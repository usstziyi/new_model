"""训练 CSFA-PMCCNN 做 EEG 句子语义分类（ds005170，39 类）。

任务（reading / imagine）的试次长度不同，必须分别训练：

    reading: pkl 2501 点 -> 首尾裁剪后 2251（默认 500 Hz）
    imagine: pkl 1651 点 -> 首尾裁剪后 1401（默认 500 Hz）
    若用 --target-sfreq 250，则分别为 1126 / 701

默认参数按 8 GB 显卡设置（bf16 混合精度 + 梯度累积，等效 batch 固定为 32、lr=2e-3、
500 Hz、50 epoch）。micro-batch 默认 imagine=8（峰值约 3.6 GB）/ reading=4（T 更长，
峰值约 5.2 GB），再按 --batch-size 自动补足 --accum-steps 到等效 32。
显存更宽裕时可直接调大 --batch-size 并相应调小 --accum-steps（如 16/2）；
显存仍不足时优先降低 --batch-size，或把 --target-sfreq 设为 250。

按天划分（每 9 个 run 为一天，共 45 个 run / 5 天）：

    训练：day 1-3（run 1-27）
    验证：day 4（run 28-36）
    测试：day 5（run 37-45）

用法::

    # 想象任务
    uv run plan_csfa_pmccnn/train.py --task imagine

    # 想象任务，限制 epoch 与 batch 数（快速试跑）
    uv run plan_csfa_pmccnn/train.py --task imagine --epochs 5 \
        --max-train-batches 20 --max-val-batches 10 --max-test-batches 10

CPU 上完整训练较慢：122 电极 × 576/1001 采样点，单个 batch 约数秒。
建议先用小样本确认流程，再放到 GPU 或分多次长时间运行。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import ConcatDataset, DataLoader

from dataset import (
    CSV_ROOT,
    ChiscoEEGDataset,
    RunBlockSampler,
    load_class_names,
)
from models import CSFAPMCCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "csfa_pmccnn"
RUNS_PER_DAY = 9
NUM_CLASSES = 39


class MarginLoss(nn.Module):
    """胶囊网络 Margin Loss（论文公式 2.24）。

    L_k = T_k * max(0, m+ - ||v_k||)^2
        + λ * (1 - T_k) * max(0, ||v_k|| - m-)^2
    """

    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, v_norm, labels):
        # v_norm: (B, num_classes) 各胶囊的模长；labels: (B,) 真实类别索引
        t = torch.zeros_like(v_norm)
        t.scatter_(1, labels.view(-1, 1), 1.0)

        present_loss = t * torch.clamp(self.m_plus - v_norm, min=0) ** 2
        absent_loss = self.lambda_ * (1 - t) * torch.clamp(v_norm - self.m_minus, min=0) ** 2
        return (present_loss + absent_loss).sum(dim=1).mean()


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runs_from_days(days, runs_per_day: int = RUNS_PER_DAY) -> list[int]:
    """把「第几天」换算成 run 编号列表：day 1 -> 1-9，day 2 -> 10-18。"""
    runs: list[int] = []
    for day in days:
        if day < 1:
            raise ValueError(f"day 必须为正整数，收到 {day}")
        start = (day - 1) * runs_per_day + 1
        runs.extend(range(start, day * runs_per_day + 1))
    return runs


def available_subjects(task: str, csv_root: Path = CSV_ROOT) -> list[str]:
    """从 CSV 文件名自动发现该任务下可用的被试。"""
    return sorted({path.name.split("_")[0] for path in Path(csv_root).glob(f"*_{task}.csv")})


def build_parts(
    subjects: list[str],
    task: str,
    runs: list[int],
    target_sfreq: float,
    normalize: bool,
    cache_size: int,
) -> list[ChiscoEEGDataset]:
    """每个被试建一个 Dataset，便于按 run 分别缓存 pkl。"""
    return [
        ChiscoEEGDataset(
            subject,
            task,
            runs=runs,
            target_sfreq=target_sfreq,
            normalize=normalize,
            cache_size=cache_size,
        )
        for subject in subjects
    ]


def concat_or_single(parts: list[ChiscoEEGDataset]):
    """单个被试时直接返回 Dataset，多被试时拼成 ConcatDataset。"""
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


def combined_run_ids(parts: list[ChiscoEEGDataset]) -> list[int]:
    """拼出与 ConcatDataset 下标一一对应的 run 编号。"""
    ids: list[int] = []
    for part in parts:
        ids.extend(int(row["run"]) for row in part.rows)
    return ids


def make_eval_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool):
    """验证 / 测试集保持 CSV 的 run 顺序，缓存命中率最高。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
    accum_steps: int = 1,
    amp_dtype=None,
    scaler=None,
):
    """跑一个 epoch，返回 (平均 loss, 样本数)。

    用累加 accum_steps 个 micro-batch 的梯度再更新一次参数，等效 batch 不变。
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    pending = 0
    use_amp = amp_dtype is not None

    def step_optimizer():
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    optimizer.zero_grad(set_to_none=True)
    for step, (alpha, beta, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        step_started = time.time()
        alpha = alpha.to(device, non_blocking=True)
        beta = beta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(alpha, beta)
            loss = criterion(outputs, labels)

        scaled_loss = loss / accum_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        loss_value = loss.item()
        total_loss += loss_value * labels.size(0)
        n_samples += labels.size(0)

        pending += 1
        print(
            f"    [step {step + 1}] loss={loss_value:.4f} batch={labels.size(0)} "
            f"samples={n_samples} accum={pending}/{accum_steps} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"({time.time() - step_started:.2f}s)",
            flush=True,
        )
        if pending == accum_steps:
            step_optimizer()
            pending = 0

    if pending:
        step_optimizer()

    if n_samples == 0:
        raise RuntimeError("训练集为空，检查 --train-days / --max-train-batches")
    return total_loss / n_samples, n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, max_batches=None, amp_dtype=None):
    """在验证 / 测试集上计算 loss、accuracy、macro-F1、weighted-F1。"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_targets = [], []
    use_amp = amp_dtype is not None

    for step, (alpha, beta, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        alpha = alpha.to(device, non_blocking=True)
        beta = beta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(alpha, beta)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_targets.append(labels.cpu())

    if n_samples == 0:
        raise RuntimeError("评估集为空，检查 --max-val-batches / --max-test-batches")

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    labels_range = list(range(num_classes))

    return {
        "loss": total_loss / n_samples,
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(
            f1_score(targets, preds, average="macro", labels=labels_range, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, preds, average="weighted", labels=labels_range, zero_division=0)
        ),
        "n_samples": n_samples,
        "preds": preds,
        "targets": targets,
    }


def save_checkpoint(path, model, task, subjects, target_sfreq, normalize, epoch, metrics):
    """保存模型与复现实验所需的配置。"""
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": 1,
            "num_classes": NUM_CLASSES,
            "task": task,
            "subjects": subjects,
            "target_sfreq": target_sfreq,
            "normalize": normalize,
            "epoch": epoch,
            "val_metrics": {k: v for k, v in metrics.items() if k not in ("preds", "targets")},
        },
        path,
    )


def report_test(
    model,
    loader,
    criterion,
    device,
    class_names: dict[int, str],
    out_dir: Path,
    max_batches=None,
    amp_dtype=None,
):
    """在测试集上评估，打印分类报告并保存混淆矩阵 / 指标。"""
    num_classes = len(class_names)
    metrics = evaluate(
        model, loader, criterion, device, num_classes, max_batches, amp_dtype
    )
    preds, targets = metrics.pop("preds"), metrics.pop("targets")

    report = classification_report(
        targets,
        preds,
        labels=list(range(num_classes)),
        target_names=[class_names[index] for index in range(num_classes)],
        digits=3,
        zero_division=0,
    )
    print("\n测试集分类报告:")
    print(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [class_names[index] for index in range(num_classes)]
    matrix = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(
        f"[Test] loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print(f"[Saved] {out_dir / 'confusion_matrix.csv'} / {out_dir / 'test_metrics.json'}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练 CSFA-PMCCNN 做 EEG 句子语义分类（39 类）。"
    )
    parser.add_argument("--task", required=True, choices=("reading", "imagine"), help="任务")
    parser.add_argument("--subjects", nargs="+", default=None, help="被试，默认自动发现")
    parser.add_argument("--train-days", nargs="+", type=int, default=[1, 2, 3], help="训练用第几天")
    parser.add_argument("--val-days", nargs="+", type=int, default=[4], help="验证用第几天")
    parser.add_argument("--test-days", nargs="+", type=int, default=[5], help="测试用第几天")
    parser.add_argument("--runs-per-day", type=int, default=RUNS_PER_DAY, help="每天 run 数")
    parser.add_argument("--epochs", type=int, default=50, help="最大 epoch 数")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="单次前向的 micro-batch size，默认 imagine=8 / reading=4（T 更长，峰值显存约 1.6 倍）",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=None,
        help="梯度累积步数，默认按 --batch-size 补足到等效 batch=32",
    )
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度（默认开启）")
    parser.add_argument("--lr", type=float, default=2e-3, help="Adam 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--patience", type=int, default=10, help="验证集 macro-F1 无提升的容忍 epoch 数")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--no-normalize", action="store_true", help="关闭逐通道 z-score")
    parser.add_argument("--cache-size", type=int, default=1, help="每个被试缓存的 run 数")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数（pkl 很大，默认 0）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="实验输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按任务 + 时间命名")
    parser.add_argument("--max-train-batches", type=int, default=None, help="每个 epoch 最多训练批数（试跑用）")
    parser.add_argument("--max-val-batches", type=int, default=None, help="验证最多批数（试跑用）")
    parser.add_argument("--max-test-batches", type=int, default=None, help="测试最多批数（试跑用）")
    parser.add_argument("--skip-test", action="store_true", help="训练结束后不跑测试集")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # micro-batch 按任务长度取默认值（imagine 1401 点、reading 2251 点），
    # 再用梯度累积把等效 batch 补回 32，保证与论文设置的一致性。
    if args.batch_size is None:
        args.batch_size = 8 if args.task == "imagine" else 4
    if args.accum_steps is None:
        args.accum_steps = max(1, 32 // args.batch_size)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Device] {device}")

    class_names = load_class_names()
    num_classes = len(class_names)

    subjects = args.subjects or available_subjects(args.task)
    if not subjects:
        raise RuntimeError(f"没有找到 task={args.task} 的 CSV：{CSV_ROOT}")
    print(f"[Subjects] {subjects}")

    train_runs = runs_from_days(args.train_days, args.runs_per_day)
    val_runs = runs_from_days(args.val_days, args.runs_per_day)
    test_runs = runs_from_days(args.test_days, args.runs_per_day)

    normalize = not args.no_normalize
    target_sfreq = args.target_sfreq

    train_parts = build_parts(subjects, args.task, train_runs, target_sfreq, normalize, args.cache_size)
    val_parts = build_parts(subjects, args.task, val_runs, target_sfreq, normalize, args.cache_size)
    test_parts = build_parts(subjects, args.task, test_runs, target_sfreq, normalize, args.cache_size)

    train_set = concat_or_single(train_parts)
    val_set = concat_or_single(val_parts)
    test_set = concat_or_single(test_parts)
    print(
        f"[Split] train={len(train_set)} val={len(val_set)} test={len(test_set)} | "
        f"runs train={train_runs[0]}-{train_runs[-1]} "
        f"val={val_runs[0]}-{val_runs[-1]} test={test_runs[0]}-{test_runs[-1]}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=RunBlockSampler(combined_run_ids(train_parts), shuffle=True, seed=args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_eval_loader(val_set, args.batch_size, args.num_workers, device.type == "cuda")
    test_loader = make_eval_loader(test_set, args.batch_size, args.num_workers, device.type == "cuda")

    model = CSFAPMCCNN(in_channels=1, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CSFAPMCCNN | classes={num_classes} | params={n_params:,}")

    # 混合精度：Ampere 及更新架构用 bf16（无需 GradScaler），否则退回 fp16 + GradScaler
    amp_dtype = None
    scaler = None
    if device.type == "cuda" and not args.no_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    print(
        f"[Precision] {amp_dtype or torch.float32} | "
        f"micro_batch={args.batch_size} accum={args.accum_steps} "
        f"effective_batch={args.batch_size * args.accum_steps}"
    )

    criterion = MarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 3)
    )

    run_name = args.run_name or f"{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    print(f"[Output] {out_dir}")

    best_f1, best_epoch, no_improve = -1.0, 0, 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_n = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.max_train_batches,
            args.accum_steps,
            amp_dtype,
            scaler,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            args.max_val_batches,
            amp_dtype,
        )
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_samples": train_n,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"lr={row['lr']:.2e} ({row['seconds']:.1f}s)"
        )

        if val_metrics["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, no_improve = val_metrics["macro_f1"], epoch, 0
            save_checkpoint(
                checkpoint_path, model, args.task, subjects, target_sfreq, normalize, epoch, val_metrics
            )
            print(f"[Best] epoch={epoch} macro_f1={best_f1:.4f} -> {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[Stop] 验证集 macro-F1 已 {args.patience} 个 epoch 未提升，提前结束")
                break

    print(f"[Done] best epoch={best_epoch} val_macro_f1={best_f1:.4f}")

    if args.skip_test:
        print("[Skip] --skip-test：不评估测试集")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = report_test(
        model, test_loader, criterion, device, class_names, out_dir, args.max_test_batches, amp_dtype
    )
    print(f"[Done] test acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
