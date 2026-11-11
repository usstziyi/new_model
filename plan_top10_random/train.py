"""训练 PMCNN 做 top-10 类 EEG 句子语义分类（随机 8:1:1 划分）。

与 ``plan_classification/train.py`` 的区别：

- 数据来自 ``datasets/top10_imagine/<subject>/class_*.pkl``，10 类样本一次性
  载入内存，标签从原始 ``type_number`` 重映射为 ``0..9``；
- 完全抛弃 day / run 概念，按类别分层随机划分 train / val / test（默认 8:1:1），
  划分由 ``--split-seed`` 固定，``test.py`` 用同一 seed 复现；
- 不再需要 ``RunBlockSampler`` / pkl LRU 缓存，``DataLoader(shuffle=True)`` 即可。

网络结构、损失、优化器、早停与学习率调度均沿用 ``plan_classification``：
``model.PMCNN`` + ``CrossEntropyLoss``，验证集 macro-F1 早停。

用法::

    # 默认 sub-01，8:1:1，随机种子 42
    uv run plan_top10_random/train.py
    uv run plan_top10_random/train.py --use-64-channels

    # 快速试跑
    uv run plan_top10_random/train.py --epochs 3 \
        --max-train-batches 20 --max-val-batches 5 --max-test-batches 5

    # 换划分种子 / 换比例
    uv run plan_top10_random/train.py --split-seed 7 --val-ratio 0.1 --test-ratio 0.1
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
from torch.utils.data import DataLoader, Subset

from dataset import TOP_ROOT, TopClassEEGDataset, split_indices
from model import PMCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "top10_random"


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(
    dataset: TopClassEEGDataset,
    train_idx,
    val_idx,
    test_idx,
    batch_size: int,
    pin_memory: bool,
):
    """训练集打乱，验证 / 测试集保持固定顺序。"""
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
    amp_dtype=None,
    scaler=None,
):
    """跑一个 epoch，返回 (平均 loss, 样本数)。"""
    model.train()
    total_loss, n_samples = 0.0, 0
    use_amp = amp_dtype is not None

    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        step_started = time.time()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_value = loss.item()
        total_loss += loss_value * labels.size(0)
        n_samples += labels.size(0)

        print(
            f"    [step {step + 1}] loss={loss_value:.4f} batch={labels.size(0)} "
            f"samples={n_samples} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"({time.time() - step_started:.2f}s)",
            flush=True,
        )

        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if n_samples == 0:
        raise RuntimeError("训练集为空，检查 --max-train-batches")
    return total_loss / n_samples, n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, max_batches=None, amp_dtype=None):
    """在验证 / 测试集上计算 loss、accuracy、macro-F1、weighted-F1。"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_targets = [], []
    use_amp = amp_dtype is not None

    for step, (inputs, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(inputs)
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


def save_checkpoint(
    path,
    model,
    model_kwargs,
    subject,
    class_ids,
    class_names,
    target_sfreq,
    normalize,
    use_64_channels,
    split_args,
    epoch,
    metrics,
):
    """保存模型与复现实验所需的配置（含划分参数与标签映射）。"""
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "subject": subject,
            "class_ids": class_ids,
            "class_names": class_names,
            "target_sfreq": target_sfreq,
            "normalize": normalize,
            "use_64_channels": use_64_channels,
            "split": split_args,
            "epoch": epoch,
            "val_metrics": {k: v for k, v in metrics.items() if k not in ("preds", "targets")},
        },
        path,
    )


def plot_confusion_heatmap(csv_path: Path, out_path: Path) -> None:
    """读取 ``confusion_matrix.csv``，用 seaborn 画一张混淆矩阵热力图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rc_context

    frame = pd.read_csv(csv_path, index_col=0)
    side = max(6.0, 0.55 * len(frame))
    with rc_context(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(side, 0.85 * side))
        sns.heatmap(
            frame,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "样本数"},
            ax=ax,
        )
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"混淆矩阵（{len(frame)} 类）")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"[Plot] 混淆矩阵热力图已保存: {out_path}")


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
    matrix_csv = out_dir / "confusion_matrix.csv"
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        matrix_csv, encoding="utf-8-sig"
    )
    plot_confusion_heatmap(matrix_csv, out_dir / "confusion_matrix.png")
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(
        f"[Test] loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print(
        f"[Saved] {matrix_csv} / {out_dir / 'confusion_matrix.png'} / "
        f"{out_dir / 'test_metrics.json'}"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="训练 PMCNN 做 top-10 类 EEG 语义分类（随机 8:1:1）。")
    # ---------- 数据 ----------
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--split-seed", type=int, default=42, help="随机划分的种子（test.py 需保持一致）")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--no-normalize", action="store_true", help="关闭逐通道 z-score")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 channels_64 列出的 64 个通道，并按该列表顺序重排（默认用 122 通道原顺序）")
    # ---------- 训练 ----------
    parser.add_argument("--epochs", type=int, default=50, help="最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32, help="单次前向的 batch size，默认 32")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度（默认开启）")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="交叉熵 label smoothing")
    parser.add_argument("--patience", type=int, default=10, help="验证集 macro-F1 无提升的容忍 epoch 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # ---------- 模型 ----------
    parser.add_argument("--e-out", type=int, default=16, help="Stem 压缩后的电极维 E_out")
    parser.add_argument("--c0", type=int, default=64, help="Stem 输出通道数")
    parser.add_argument("--stem-k", type=int, default=7, help="Stem 时间卷积核长")
    parser.add_argument("--dropout", type=float, default=0.5, help="分类头 dropout")
    parser.add_argument("--use-electrode-attn", action="store_true", help="在 E 轴上做注意力池化")
    # ---------- 输出 ----------
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="实验输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    parser.add_argument("--max-train-batches", type=int, default=None, help="每个 epoch 最多训练批数（试跑用）")
    parser.add_argument("--max-val-batches", type=int, default=None, help="验证最多批数（试跑用）")
    parser.add_argument("--max-test-batches", type=int, default=None, help="测试最多批数（试跑用）")
    parser.add_argument("--skip-test", action="store_true", help="训练结束后不跑测试集")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Device] {device}")

    normalize = not args.no_normalize
    dataset = TopClassEEGDataset(
        subject=args.subject,
        root=args.data_root,
        target_sfreq=args.target_sfreq,
        normalize=normalize,
        use_64_channels=args.use_64_channels,
    )
    num_classes = len(dataset.class_ids)
    class_names = dict(enumerate(dataset.class_names))
    print(
        f"[Classes] {num_classes} | type_number -> label: "
        f"{dict(zip(range(num_classes), dataset.class_ids))}"
    )

    train_idx, val_idx, test_idx = split_indices(
        dataset.labels, args.val_ratio, args.test_ratio, args.split_seed
    )
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = np.bincount(dataset.labels[idx], minlength=num_classes)
        print(f"[Split] {name:<5} {len(idx):>5} 条 | 每类 {counts.min()}-{counts.max()}")

    train_loader, val_loader, test_loader = make_loaders(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        args.batch_size,
        pin_memory=device.type == "cuda",
    )
    n_channels = dataset.n_channels
    print(
        f"[Channels] {n_channels} "
        f"({'channels_64 顺序' if args.use_64_channels else 'fif 原顺序'})"
    )

    model_kwargs = dict(
        num_classes=num_classes,
        E_in=n_channels,
        E_out=args.e_out,
        C0=args.c0,
        stem_k=args.stem_k,
        dropout=args.dropout,
        use_electrode_attn=args.use_electrode_attn,
    )
    model = PMCNN(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] PMCNN | classes={num_classes} | E_in={n_channels} E_out={args.e_out} "
        f"| params={n_params:,}"
    )

    # 混合精度：Ampere 及更新架构用 bf16（无需 GradScaler），否则退回 fp16 + GradScaler
    amp_dtype = None
    scaler = None
    if device.type == "cuda" and not args.no_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    print(f"[Precision] {amp_dtype or torch.float32} | batch={args.batch_size}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 3)
    )

    run_name = args.run_name or f"top10_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    print(f"[Output] {out_dir}")

    split_args = {
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.split_seed,
    }

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
                checkpoint_path,
                model,
                model_kwargs,
                args.subject,
                dataset.class_ids,
                dataset.class_names,
                args.target_sfreq,
                normalize,
                args.use_64_channels,
                split_args,
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
