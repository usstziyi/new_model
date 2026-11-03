"""训练单分支 CNN + MLP 做 EEG 句子语义分类（ds005170，39 类）。

相比 CSFA-PMCCNN：

- 单分支：α / β 等频段在通道维拼接后只走一条卷积主干（``--bands`` 控制）；
- 去掉融合后端与胶囊层，分类头为「普通 3×3 卷积 -> GAP -> MLP」；
- 损失函数为交叉熵 ``nn.CrossEntropyLoss``。

# 想象任务（imagine）的试次长度不同，必须分别训练：

    reading: pkl 2501 点 -> 首尾裁剪后 2251（默认 500 Hz）
    imagine: pkl 1651 点 -> 首尾裁剪后 1401（默认 500 Hz）
    若用 --target-sfreq 250，则分别为 1126 / 701

数据划分：按 run 随机划分（同一个 run 的试次只会出现在一侧，避免信息泄漏），
默认训练 80% / 验证 20%，不设测试集。划分由 --seed 固定，因此可复现。

用法::

    # 想象任务，默认 α+β 双通道输入，按 80/20 划分 run
    uv run plan_single_cnn_mlp/train.py --task imagine

    # 严格单频段（单通道）输入
    uv run plan_single_cnn_mlp/train.py --task imagine --bands alpha

    # 想象任务，限制 epoch 与 batch 数（快速试跑）
    uv run plan_single_cnn_mlp/train.py --task imagine --epochs 5 \
        --max-train-batches 20 --max-val-batches 10
"""

from __future__ import annotations

import argparse
import csv
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
    DEFAULT_BANDS,
    ChiscoEEGDataset,
    RunBlockSampler,
    list_runs,
    load_class_names,
)
from models import SingleBranchCNNMLP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "single_cnn_mlp"
NUM_CLASSES = 39


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def available_runs(subjects: list[str], task: str) -> list[int]:
    """收集该任务下所有被试 CSV 中出现过的 run 编号。"""
    runs: set[int] = set()
    for subject in subjects:
        runs.update(list_runs(subject, task))
    return sorted(runs)


def split_runs(runs: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """按 run 随机划分训练 / 验证集。

    以 run 为最小单位划分，保证同一个 run 的试次不会同时进入训练和验证，
    避免同一段记录内的相关性造成验证指标虚高。默认 20% 的 run 作为验证集，
    划分结果由 ``seed`` 决定，重复运行可复现。
    """
    runs = sorted({int(run) for run in runs})
    if len(runs) < 2:
        raise ValueError(f"至少需要 2 个 run 才能划分，当前只有 {len(runs)} 个")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio 需在 (0, 1) 内，收到 {val_ratio}")

    n_val = int(round(len(runs) * val_ratio))
    n_val = min(max(n_val, 1), len(runs) - 1)
    order = np.random.default_rng(seed).permutation(len(runs))
    val_runs = sorted(runs[i] for i in order[:n_val])
    train_runs = sorted(runs[i] for i in order[n_val:])
    return train_runs, val_runs


def available_subjects(task: str, csv_root: Path = CSV_ROOT) -> list[str]:
    """从 CSV 文件名自动发现该任务下可用的被试。"""
    return sorted({path.name.split("_")[0] for path in Path(csv_root).glob(f"*_{task}.csv")})


def build_parts(
    subjects: list[str],
    task: str,
    runs: list[int],
    bands: tuple[str, ...],
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
            bands=bands,
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
    """验证集保持 CSV 的 run 顺序，缓存命中率最高。"""
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
    step_log_path: Path | None = None,
    epoch: int = 1,
):
    """跑一个 epoch，返回 (平均 loss, 样本数)。

    ``step_log_path`` 不为空时，把每个 step 的记录增量追加到该 CSV
    （文件已存在则接着写，不存在则先写表头），每写一行立即 flush。
    """
    model.train()
    total_loss, n_samples = 0.0, 0

    log_handle = None
    log_writer = None
    if step_log_path is not None:
        step_log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not step_log_path.exists() or step_log_path.stat().st_size == 0
        log_handle = step_log_path.open("a", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_handle,
            fieldnames=["epoch", "step", "loss", "batch", "samples", "lr", "seconds"],
        )
        if new_file:
            log_writer.writeheader()
            log_handle.flush()

    try:
        for step, (inputs, labels) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            step_started = time.time()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value * labels.size(0)
            n_samples += labels.size(0)
            elapsed = time.time() - step_started

            print(
                f"    [step {step + 1}] loss={loss_value:.4f} batch={labels.size(0)} "
                f"samples={n_samples} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"({elapsed:.2f}s)",
                flush=True,
            )

            if log_writer is not None:
                log_writer.writerow(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": f"{loss_value:.6f}",
                        "batch": labels.size(0),
                        "samples": n_samples,
                        "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                        "seconds": f"{elapsed:.3f}",
                    }
                )
                log_handle.flush()
    finally:
        if log_handle is not None:
            log_handle.close()

    if n_samples == 0:
        raise RuntimeError("训练集为空，检查 --train-days / --max-train-batches")
    return total_loss / n_samples, n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, max_batches=None):
    """在验证集上计算 loss、accuracy、macro-F1、weighted-F1。"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_targets = [], []

    for step, (inputs, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_targets.append(labels.cpu())

    if n_samples == 0:
        raise RuntimeError("验证集为空，检查 --max-val-batches")

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


def save_checkpoint(
    path,
    model,
    task,
    subjects,
    bands,
    target_sfreq,
    normalize,
    train_runs,
    val_runs,
    epoch,
    metrics,
):
    """保存模型与复现实验所需的配置（含 train / val 的 run 划分）。"""
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": len(bands),
            "num_classes": model.classifier[-1].out_features,
            "task": task,
            "subjects": subjects,
            "bands": bands,
            "target_sfreq": target_sfreq,
            "normalize": normalize,
            "train_runs": list(train_runs),
            "val_runs": list(val_runs),
            "epoch": epoch,
            "val_metrics": {k: v for k, v in metrics.items() if k not in ("preds", "targets")},
        },
        path,
    )


def report_metrics(
    model,
    loader,
    criterion,
    device,
    class_names: dict[int, str],
    out_dir: Path,
    split_name: str = "val",
    max_batches=None,
):
    """在验证集上评估，打印分类报告并保存混淆矩阵 / 指标。"""
    num_classes = len(class_names)
    metrics = evaluate(model, loader, criterion, device, num_classes, max_batches)
    preds, targets = metrics.pop("preds"), metrics.pop("targets")

    report = classification_report(
        targets,
        preds,
        labels=list(range(num_classes)),
        target_names=[class_names[index] for index in range(num_classes)],
        digits=3,
        zero_division=0,
    )
    print(f"\n{split_name} 分类报告:")
    print(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [class_names[index] for index in range(num_classes)]
    matrix = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    matrix_path = out_dir / f"{split_name}_confusion_matrix.csv"
    metrics_path = out_dir / f"{split_name}_metrics.json"
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        matrix_path, encoding="utf-8-sig"
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(
        f"[{split_name}] loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print(f"[Saved] {matrix_path} / {metrics_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练单分支 CNN + MLP 做 EEG 句子语义分类（39 类）。"
    )
    parser.add_argument("--task", required=True, choices=("reading", "imagine"), help="任务")
    parser.add_argument("--subjects", nargs="+", default=None, help="被试，默认自动发现")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS), choices=("alpha", "beta", "gamma"), help="输入频段，按顺序在通道维拼接；默认 alpha beta（2 通道）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集占全部 run 的比例，默认 0.2")
    parser.add_argument("--epochs", type=int, default=50, help="最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
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
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

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

    bands = tuple(args.bands)
    all_runs = available_runs(subjects, args.task)
    train_runs, val_runs = split_runs(all_runs, args.val_ratio, args.seed)
    print(
        f"[Split] runs={len(all_runs)} -> train={len(train_runs)} "
        f"val={len(val_runs)} (val_ratio={args.val_ratio:.0%}, seed={args.seed})"
    )
    print(f"[Runs] train={train_runs}")
    print(f"[Runs] val={val_runs}")

    normalize = not args.no_normalize
    target_sfreq = args.target_sfreq

    train_parts = build_parts(
        subjects, args.task, train_runs, bands, target_sfreq, normalize, args.cache_size
    )
    val_parts = build_parts(
        subjects, args.task, val_runs, bands, target_sfreq, normalize, args.cache_size
    )

    train_set = concat_or_single(train_parts)
    val_set = concat_or_single(val_parts)
    total_samples = len(train_set) + len(val_set)
    print(
        f"[Split] train={len(train_set)} ({len(train_set) / total_samples:.1%}) "
        f"val={len(val_set)} ({len(val_set) / total_samples:.1%})"
    )
    print(f"[Input] bands={bands} channels={len(bands)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=RunBlockSampler(combined_run_ids(train_parts), shuffle=True, seed=args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_eval_loader(val_set, args.batch_size, args.num_workers, device.type == "cuda")

    model = SingleBranchCNNMLP(in_channels=len(bands), num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] SingleBranchCNNMLP | classes={num_classes} | params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
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
            step_log_path=out_dir / "step_log.csv",
            epoch=epoch,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, num_classes, args.max_val_batches
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
                checkpoint_path,
                model,
                args.task,
                subjects,
                bands,
                target_sfreq,
                normalize,
                train_runs,
                val_runs,
                epoch,
                val_metrics,
            )
            print(f"[Best] epoch={epoch} macro_f1={best_f1:.4f} -> {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[Stop] 验证集 macro-F1 已 {args.patience} 个 epoch 未提升，提前结束")
                break

    print(f"[Done] best epoch={best_epoch} val_macro_f1={best_f1:.4f}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics = report_metrics(
        model,
        val_loader,
        criterion,
        device,
        class_names,
        out_dir,
        split_name="val",
        max_batches=args.max_val_batches,
    )
    print(f"[Done] val acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
