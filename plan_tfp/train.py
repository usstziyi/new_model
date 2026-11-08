"""TFP-CTNet 五折训练与评估（Chisco 中文词语级，ACC / macro-F1）。

对应论文 4.3.2 的评估协议：被试内分层 5 折，每折约 80% 训练 / 20% 测试。
论文没有单独划验证集，这里从每折的训练部分再分层留出 ``--val-ratio``（默认
10%）用于早停与选模型，保证测试集只参与最终报告。

用法::

    uv run plan_tfp/train.py
    uv run plan_tfp/train.py --epochs 2 --max-train-batches 5 --max-test-batches 2

输出::

    outputs/tfp_ctnet/<run>/folds.csv     每折 best_epoch / test_acc / test_f1
    outputs/tfp_ctnet/<run>/summary.json  五折均值与标准差
    outputs/tfp_ctnet/<run>/fold{k}.pt    该折最优权重
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import TFPEEGDataset, stratified_folds
from tfp_ctnet_reconstruction import TFPCTNet, TFPCTNetConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "tfp_ctnet"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_precision(device: torch.device, use_amp: bool = True):
    """按设备自动选择精度。

    - CUDA 且支持 bf16（Ampere 及更新）：``bfloat16`` 自动混合精度，无需 GradScaler；
    - CUDA 但只支持 fp16：``float16`` + ``GradScaler``；
    - CPU / MPS / 关闭 AMP：纯 ``float32``（返回 ``None``）。

    返回 ``(amp_dtype, scaler)``，两者都为 ``None`` 表示全精度。
    """
    if not use_amp or device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """线性 warmup + cosine 衰减，按 step 更新。"""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model, loader, optimizer, device, max_batches=None, *,
    epoch=None, epochs=None, amp_dtype=None, scaler=None,
    scheduler=None, grad_clip=None, label_smoothing: float = 0.0,
) -> dict:
    """跑一个 epoch，返回平均交叉熵与训练准确率；每个 step 打印一行。"""
    model.train()
    total, n_correct, n_samples = 0.0, 0, 0
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    tag = f"[epoch {epoch:03d}/{epochs}]" if epoch is not None else "[epoch]"

    for step, (inputs, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        started = time.time()
        inputs = inputs.squeeze(1).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(inputs)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += loss.item() * labels.size(0)
        n_correct += (logits.argmax(dim=1) == labels).sum().item()
        n_samples += labels.size(0)
        print(
            f"  {tag}[step {step + 1:02d}/{n_batches}] "
            f"loss={loss.item():.4f} avg={total / n_samples:.4f} "
            f"acc={n_correct / n_samples:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"batch={labels.size(0)} ({time.time() - started:.2f}s)",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
    if n_samples == 0:
        raise RuntimeError("训练集为空，检查 --max-train-batches")
    return {"loss": total / n_samples, "acc": n_correct / n_samples}


@torch.no_grad()
def evaluate(model, loader, device, n_classes, max_batches=None, amp_dtype=None) -> dict:
    """返回 ``{loss, acc, f1}``，F1 用 macro 平均以抵消类别不均衡。"""
    model.eval()
    total, n_samples = 0.0, 0
    preds, targets = [], []
    for step, (inputs, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs = inputs.squeeze(1).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(inputs)
        loss = F.cross_entropy(logits, labels.to(device, non_blocking=True))

        total += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        preds.append(logits.argmax(dim=1).float().cpu())
        targets.append(labels)

    if n_samples == 0:
        raise RuntimeError("评估集为空，检查 --max-test-batches")
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return {
        "loss": total / n_samples,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro",
                             labels=list(range(n_classes)), zero_division=0)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TFP-CTNet 五折训练（ACC / macro-F1）。")
    # ---------- 数据 ----------
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--top-k", type=int, default=10, help="datasets/top<k>_imagine 的 k")
    parser.add_argument("--data-root", type=Path, default=None, help="覆盖默认数据根目录")
    parser.add_argument("--no-normalize", action="store_true", help="关闭逐通道 z-score")
    # ---------- 拓扑先验 ----------
    parser.add_argument("--sigma-mode", choices=("nn", "median"), default="nn", help="电极邻接高斯核 σ 的取法：nn=最近邻距离（默认），median=所有距离中位数（消融基线）")
    parser.add_argument("--nn-factor", type=float, default=2.0, help="--sigma-mode nn 时 σ = nn_factor × 最近邻距离中位数")
    # ---------- 划分 ----------
    parser.add_argument("--folds", type=int, default=5, help="折数")
    parser.add_argument("--only-fold", type=int, default=None, help="只跑指定的第 N 折（划分仍按 --folds），用于快速对照")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="每折从训练部分留出的验证比例")
    parser.add_argument("--split-seed", type=int, default=0, help="折划分种子")
    # ---------- 模型容量 ----------
    parser.add_argument("--d-model", type=int, default=64, help="Transformer 隐层维度")
    parser.add_argument("--n-layers", type=int, default=3, help="Transformer 编码层数")
    parser.add_argument("--conv-dropout", type=float, default=0.2, help="多尺度卷积 dropout")
    parser.add_argument("--transformer-dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="门控残差 MLP dropout")
    # ---------- 训练 ----------
    parser.add_argument("--epochs", type=int, default=100, help="每折最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 峰值学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="交叉熵 label smoothing")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="线性 warmup 的 epoch 数")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值；<=0 表示关闭")
    parser.add_argument("--patience", type=int, default=15, help="验证 loss 无改善的容忍 epoch 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-amp", action="store_true", help="关闭自动混合精度（默认按设备自动选择：CUDA→bf16/fp16，CPU→fp32）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数；数据全在内存里，Windows 下 >0 会复制整份数据集，建议保持 0")
    # ---------- 输出 ----------
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="实验输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    parser.add_argument("--max-train-batches", type=int, default=None, help="每 epoch 最多训练批数（试跑用）")
    parser.add_argument("--max-test-batches", type=int, default=None, help="评估最多批数（试跑用）")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds 至少为 2")
    if args.only_fold is not None and not 1 <= args.only_fold <= args.folds:
        parser.error(f"--only-fold 必须在 1..{args.folds} 之间")
    if args.device and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("指定了 --device cuda，但当前 torch 不支持 CUDA")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")

    amp_dtype, scaler = pick_precision(device, use_amp=not args.no_amp)
    print(
        "[Precision] "
        + (f"{str(amp_dtype).replace('torch.', '')} autocast"
           + (" + GradScaler" if scaler is not None else "")
           if amp_dtype is not None else "float32 (全精度)")
    )

    dataset = TFPEEGDataset(
        subject=args.subject,
        top_k=args.top_k,
        root=args.data_root,
        normalize=not args.no_normalize,
        sigma_mode=args.sigma_mode,
        nn_factor=args.nn_factor,
    )
    n_classes = dataset.n_classes
    eff = 1.0 / (dataset.adjacency ** 2).sum(1)
    print(
        f"[Adjacency] sigma={args.sigma_mode}"
        + (f"×{args.nn_factor:g}NN" if args.sigma_mode == "nn" else "")
        + f" | 有效邻居 {eff.median():.0f}/{len(dataset.layout.names)}"
    )
    pin_memory = device.type == "cuda"
    print(
        f"[Classes] {n_classes} | type_number -> label: "
        f"{dict(enumerate(dataset.class_ids))}"
    )

    config = TFPCTNetConfig(
        n_times=dataset.data.shape[-1],
        n_classes=n_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        conv_dropout=args.conv_dropout,
        transformer_dropout=args.transformer_dropout,
        mlp_dropout=args.mlp_dropout,
    )
    run_name = args.run_name or f"tfp_{args.subject}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] {out_dir}")

    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(
        stratified_folds(dataset.labels, args.folds, args.split_seed), start=1
    ):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        # 从训练部分再切出一份验证集，测试集全程不参与选模型
        fit_idx, val_idx = train_test_split(
            train_idx,
            test_size=args.val_ratio,
            stratify=dataset.labels[train_idx],
            random_state=args.seed,
        )
        loaders = {
            "train": DataLoader(Subset(dataset, fit_idx), batch_size=args.batch_size,
                                shuffle=True, drop_last=True, num_workers=args.num_workers,
                                pin_memory=pin_memory),
            "val": DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin_memory),
            "test": DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=pin_memory),
        }
        counts = np.bincount(dataset.labels[test_idx], minlength=n_classes)
        print(
            f"\n[Fold {fold}/{args.folds}] fit={len(fit_idx)} val={len(val_idx)} "
            f"test={len(test_idx)} | 测试集每类 {counts.min()}-{counts.max()}"
        )

        model = TFPCTNet(config, adjacency=dataset.adjacency).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        steps_per_epoch = (
            len(loaders["train"]) if args.max_train_batches is None
            else min(len(loaders["train"]), args.max_train_batches)
        )
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = max(1, min(total_steps, steps_per_epoch * args.warmup_epochs))
        scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
        grad_clip = None if args.grad_clip <= 0 else args.grad_clip
        if not rows:
            print(
                f"  [Model] d_model={args.d_model} layers={args.n_layers} | "
                f"参数 {sum(p.numel() for p in model.parameters()):,}"
            )
            print(
                f"  [Schedule] {steps_per_epoch} step/epoch | warmup {warmup_steps} 步 "
                f"-> cosine 至 {total_steps} 步 | lr={args.lr:g} clip={args.grad_clip:g}"
            )

        best_loss = float("inf")
        best_f1, best_acc, best_epoch, best_state, no_improve = -1.0, 0.0, 0, None, 0
        best_train_loss, best_train_acc = 0.0, 0.0
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            train = train_one_epoch(
                model, loaders["train"], optimizer, device, args.max_train_batches,
                epoch=epoch, epochs=args.epochs,
                amp_dtype=amp_dtype, scaler=scaler,
                scheduler=scheduler, grad_clip=grad_clip,
                label_smoothing=args.label_smoothing,
            )
            val = evaluate(
                model, loaders["val"], device, n_classes,
                args.max_test_batches, amp_dtype,
            )
            print(
                f"  [epoch {epoch:03d}/{args.epochs}] 汇总 "
                f"train_loss={train['loss']:.4f} train_acc={train['acc']:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
                f"val_f1={val['f1']:.4f}"
            )

            # 早停看 val_loss：验证集只有几百条，macro-F1 抖动太大，容易被噪声选中过拟合点
            if val["loss"] < best_loss - 1e-4:
                best_loss = val["loss"]
                best_f1, best_acc, best_epoch = val["f1"], val["acc"], epoch
                best_train_loss, best_train_acc = train["loss"], train["acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  [Stop] 验证 loss 已 {args.patience} 个 epoch 未改善")
                    break

        model.load_state_dict(best_state)
        test = evaluate(
            model, loaders["test"], device, n_classes,
            args.max_test_batches, amp_dtype,
        )
        seconds = time.time() - started
        rows.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "train_loss": best_train_loss,
            "train_acc": best_train_acc,
            "val_loss": best_loss,
            "val_acc": best_acc,
            "val_f1": best_f1,
            "test_acc": test["acc"],
            "test_f1": test["f1"],
            "test_loss": test["loss"],
            "seconds": seconds,
        })
        torch.save({"model_state": best_state, "config": vars(config),
                    "class_ids": dataset.class_ids, "fold": fold},
                   out_dir / f"fold{fold}.pt")
        print(
            f"  [Fold {fold}] best_epoch={best_epoch} "
            f"test_acc={test['acc']:.4f} test_f1={test['f1']:.4f} ({seconds:.1f}s)"
        )
        pd.DataFrame(rows).to_csv(out_dir / "folds.csv", index=False, encoding="utf-8-sig")

    accs = np.array([row["test_acc"] for row in rows])
    f1s = np.array([row["test_f1"] for row in rows])
    summary = {
        "subject": args.subject,
        "n_classes": n_classes,
        "n_folds": len(rows),
        "epochs": args.epochs,
        "sigma_mode": args.sigma_mode,
        "nn_factor": args.nn_factor,
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        f"\n[Summary] {args.subject} {n_classes} 类 {len(rows)} 折 | "
        f"ACC {100 * summary['acc_mean']:.2f} ± {100 * summary['acc_std']:.2f} | "
        f"F1 {100 * summary['f1_mean']:.2f} ± {100 * summary['f1_std']:.2f}"
    )


if __name__ == "__main__":
    main()
