"""EEG-文本嵌入空间对齐（top-10 类数据，PMCNN + 冻结 LaBSE）。

与 ``plan_top10_random/train.py`` 的区别：

- 目标从语义分类改为跨模态对齐：EEG 经 PMCNN 主干 + 投影头得到 768 维 L2
  归一化向量，与冻结 LaBSE 的句向量做 InfoNCE；
- 负样本使用**全量文本库**（2459 个唯一句子），而不是 batch 内负样本；
- 文本向量在训练开始前一次性算好（约 7.5 MB），训练循环里只做查表；
- 评估改为跨模态检索（R@1/5/10、MRR、median rank），并用 kNN 分类做语义结构
  诊断；早停默认看验证集 MRR；
- 分类头保留但默认不参与训练（``--ce-weight`` 默认 0，纯对齐）。

数据与划分沿用 ``plan_top10_random``：``datasets/top10_imagine/<subject>``
一次性载入内存，按类别分层随机 8:1:1。

用法::

    # 默认 sub-01，全量文本库负样本，纯对齐
    uv run plan_top10_align/train.py

    # 快速试跑
    uv run plan_top10_align/train.py --epochs 2 \
        --max-train-batches 5 --max-val-batches 2 --skip-test

    # 对齐 + 小权重辅助 CE
    uv run plan_top10_align/train.py --ce-weight 0.1
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from dataset import TOP_ROOT, TopClassEEGDataset, split_indices
from model import PMCNN
from text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "top10_align"


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(dataset, train_idx, val_idx, test_idx, batch_size, pin_memory):
    """训练集打乱，验证 / 测试集保持固定顺序。

    验证 / 测试必须 ``shuffle=False``：检索评估要求 ``z_eeg`` 与文本库索引
    一一对应。
    """
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


def align_loss(z_eeg, text_bank, target_index, temperature):
    """全量文本库负样本的 InfoNCE。

    Args:
        z_eeg: ``(B, D)``，已 L2 归一化的 EEG 向量。
        text_bank: ``(N, D)``，全量唯一句子的冻结 LaBSE 向量。
        target_index: ``(B,)``，每个样本正样本在 ``text_bank`` 中的下标。
        temperature: 温度 tau，越小越尖锐。
    """
    logits = z_eeg @ text_bank.t() / temperature        # (B, N)
    return F.cross_entropy(logits, target_index)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    text_bank,
    temperature,
    ce_weight=0.0,
    label_smoothing=0.0,
    max_batches=None,
    amp_dtype=None,
    scaler=None,
):
    """跑一个 epoch，返回 (平均对齐 loss, 平均分类 loss, 样本数)。"""
    model.train()
    total_align, total_cls, n_samples = 0.0, 0.0, 0
    use_amp = amp_dtype is not None

    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, labels, indices) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        step_started = time.time()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            features = model.features(inputs)
            z_eeg = model.embed(features)
            logits = model.classify(features) if ce_weight > 0 else None

        # 对比损失用 fp32 算，避免 bf16 下 768 维点积的精度损失
        loss = align_loss(z_eeg.float(), text_bank, indices, temperature)
        align_value = loss.item()
        if logits is not None:
            loss_cls = F.cross_entropy(
                logits.float(), labels, label_smoothing=label_smoothing
            )
            loss = loss + ce_weight * loss_cls
            cls_value = loss_cls.item()
            total_cls += cls_value * labels.size(0)
        else:
            cls_value = 0.0

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_align += align_value * labels.size(0)
        n_samples += labels.size(0)

        elapsed = time.time() - step_started
        cls_text = f"cls={cls_value:.4f} " if ce_weight > 0 else ""
        print(
            f"    [step {step + 1}] align={align_value:.4f} {cls_text}"
            f"batch={labels.size(0)} samples={n_samples} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} ({elapsed:.2f}s)",
            flush=True,
        )

    if n_samples == 0:
        raise RuntimeError("训练集为空，检查 --max-train-batches")
    return total_align / n_samples, total_cls / n_samples, n_samples


@torch.no_grad()
def extract_embeddings(model, loader, device, max_batches=None, amp_dtype=None):
    """返回 ``(z_eeg, indices, labels)``，三者顺序与 loader 输出一致。"""
    model.eval()
    use_amp = amp_dtype is not None
    zs, idxs, labels = [], [], []

    for step, (inputs, targets, indices) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z = model(inputs)
        zs.append(z.float().cpu())
        idxs.append(indices.cpu())
        labels.append(targets.cpu())

    if not zs:
        raise RuntimeError("评估集为空，检查 --max-val-batches / --max-test-batches")
    return torch.cat(zs), torch.cat(idxs), torch.cat(labels)


def retrieval_metrics(z_query, bank, target_index, temperature=0.07):
    """跨模态检索指标：R@1/5/10、MRR、median rank，以及检索式 CE loss。

    Args:
        z_query: ``(Q, D)`` 查询向量（EEG）。
        bank: ``(C, D)`` 候选文本向量。
        target_index: ``(Q,)`` 每条查询正样本在 ``bank`` 中的下标。
    """
    z = F.normalize(z_query.detach().float().cpu(), dim=-1)
    bank = F.normalize(bank.detach().float().cpu(), dim=-1)
    target_index = target_index.detach().cpu()
    sim = z @ bank.t()                                   # (Q, C)
    positive = sim.gather(1, target_index.view(-1, 1)).squeeze(1)
    ranks = (sim > positive.unsqueeze(1)).sum(dim=1) + 1  # (Q,)
    ranks_f = ranks.float()

    return {
        "n": int(len(ranks)),
        "candidates": int(sim.size(1)),
        "r1": float((ranks <= 1).float().mean()),
        "r5": float((ranks <= 5).float().mean()),
        "r10": float((ranks <= 10).float().mean()),
        "mrr": float((1.0 / ranks_f).mean()),
        "median_rank": float(ranks_f.median()),
        "loss": float(F.cross_entropy(sim / temperature, target_index)),
    }


def knn_diagnostic(z_train, y_train, z_eval, y_eval, k=5, num_classes=10):
    """用训练集 EEG 向量做 kNN 分类，检查对齐空间是否保留语义结构。

    这不是训练目标，仅作诊断：若 kNN 准确率仍接近随机水平，说明 embedding
    虽然可能满足对齐损失，但没有形成可分的语义簇。
    """
    from sklearn.neighbors import KNeighborsClassifier

    k = max(1, min(k, len(z_train)))
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    clf.fit(z_train.numpy(), y_train.numpy())
    preds = clf.predict(z_eval.numpy())
    targets = y_eval.numpy()

    return {
        "k": int(k),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(
            f1_score(targets, preds, average="macro",
                     labels=list(range(num_classes)), zero_division=0)
        ),
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
    text_model,
    embed_dim,
    temperature,
    select_metric,
    epoch,
    score,
    metrics,
):
    """保存模型与复现实验所需的全部配置。"""
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
            "text_model": text_model,
            "embed_dim": embed_dim,
            "temperature": temperature,
            "select_metric": select_metric,
            "epoch": epoch,
            "score": score,
            "val_metrics": metrics,
        },
        path,
    )


def report_split(
    model,
    loader,
    text_bank,
    out_dir,
    split_name,
    temperature,
    device,
    num_classes,
    max_batches=None,
    amp_dtype=None,
    train_gallery=None,
    knn_k=5,
):
    """在指定划分上算检索指标（本划分内部 + 全量文本库）与 kNN 诊断。"""
    z_eeg, indices, labels = extract_embeddings(
        model, loader, device, max_batches, amp_dtype
    )

    # 1) 只在当前划分内部检索：难度较低，适合看训练是否有效
    in_split = retrieval_metrics(
        z_eeg, text_bank[indices], torch.arange(len(indices)), temperature
    )
    # 2) 在全量文本库（2459 句）中检索：更接近真实使用场景
    full_bank = retrieval_metrics(z_eeg, text_bank, indices, temperature)

    report = {
        "split": split_name,
        "retrieval_in_split": in_split,
        "retrieval_full_bank": full_bank,
    }

    if train_gallery is not None:
        z_train, y_train = train_gallery
        report["knn"] = knn_diagnostic(
            z_train, y_train, z_eeg, labels, k=knn_k, num_classes=num_classes
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_name}_retrieval.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"\n[{split_name}] 检索报告（{in_split['n']} 条查询）")
    print(
        f"  划分内检索 ({in_split['candidates']} 候选): "
        f"R@1={in_split['r1']:.4f} R@5={in_split['r5']:.4f} "
        f"R@10={in_split['r10']:.4f} MRR={in_split['mrr']:.4f} "
        f"median={in_split['median_rank']:.1f} loss={in_split['loss']:.4f}"
    )
    print(
        f"  全量库检索 ({full_bank['candidates']} 候选): "
        f"R@1={full_bank['r1']:.4f} R@5={full_bank['r5']:.4f} "
        f"R@10={full_bank['r10']:.4f} MRR={full_bank['mrr']:.4f} "
        f"median={full_bank['median_rank']:.1f} loss={full_bank['loss']:.4f}"
    )
    if "knn" in report:
        print(
            f"  kNN 诊断 (k={report['knn']['k']}): "
            f"acc={report['knn']['accuracy']:.4f} "
            f"macro_f1={report['knn']['macro_f1']:.4f}"
        )
    print(f"[Saved] {path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG-文本嵌入对齐（top-10 类，PMCNN + 冻结 LaBSE）。"
    )
    # ---------- 数据 ----------
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--split-seed", type=int, default=42, help="随机划分种子（test.py 需保持一致）")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--no-normalize", action="store_true", help="关闭逐通道 z-score")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 channels_64 的 64 个通道")
    # ---------- 对齐 ----------
    parser.add_argument("--text-model", default="sentence-transformers/LaBSE", help="LaBSE 模型名或本地路径")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE 温度 tau")
    parser.add_argument("--proj-hidden", type=int, default=512, help="投影头隐层维度")
    parser.add_argument("--ce-weight", type=float, default=0.0, help="辅助分类 CE 权重（0 表示纯对齐）")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="辅助 CE 的 label smoothing")
    # ---------- 训练 ----------
    parser.add_argument("--epochs", type=int, default=50, help="最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度（默认开启）")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--patience", type=int, default=10, help="验证指标无提升的容忍 epoch 数")
    parser.add_argument("--select-metric", default="mrr", choices=("mrr", "r1", "r5"), help="早停与保存 best 使用的验证指标")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # ---------- 模型 ----------
    parser.add_argument("--e-out", type=int, default=16, help="Stem 压缩后的电极维 E_out")
    parser.add_argument("--c0", type=int, default=64, help="Stem 输出通道数")
    parser.add_argument("--stem-k", type=int, default=7, help="Stem 时间卷积核长")
    parser.add_argument("--dropout", type=float, default=0.5, help="投影头 dropout")
    parser.add_argument("--use-electrode-attn", action=argparse.BooleanOptionalAction, default=True, help="在 E 轴上做注意力池化（默认开启，--no-use-electrode-attn 关闭）")
    # ---------- 诊断 / 输出 ----------
    parser.add_argument("--knn-k", type=int, default=5, help="kNN 诊断的近邻数")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="实验输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    parser.add_argument("--max-train-batches", type=int, default=None, help="每个 epoch 最多训练批数（试跑用）")
    parser.add_argument("--max-val-batches", type=int, default=None, help="验证最多批数（试跑用）")
    parser.add_argument("--max-test-batches", type=int, default=None, help="测试最多批数（试跑用）")
    parser.add_argument("--skip-test", action="store_true", help="训练结束后不评估测试集")
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
    print(
        f"[Classes] {num_classes} | type_number -> label: "
        f"{dict(zip(range(num_classes), dataset.class_ids))}"
    )

    if not all(text.strip() for text in dataset.texts):
        raise RuntimeError("数据中有空句子，无法做文本对齐")

    train_idx, val_idx, test_idx = split_indices(
        dataset.labels, args.val_ratio, args.test_ratio, args.split_seed
    )
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = np.bincount(dataset.labels[idx], minlength=num_classes)
        print(f"[Split] {name:<5} {len(idx):>5} 条 | 每类 {counts.min()}-{counts.max()}")

    train_loader, val_loader, test_loader = make_loaders(
        dataset, train_idx, val_idx, test_idx, args.batch_size,
        pin_memory=device.type == "cuda",
    )
    n_channels = dataset.n_channels
    print(
        f"[Channels] {n_channels} "
        f"({'channels_64 顺序' if args.use_64_channels else 'fif 原顺序'})"
    )

    # ---------- 预计算全量文本向量库 ----------
    print(f"[Text] 正在用 {args.text_model} 编码 {len(dataset.texts)} 个句子 ...", flush=True)
    text_started = time.time()
    text_encoder = LaBSETextEncoder(args.text_model)
    text_bank = text_encoder(dataset.texts).to(device)   # (N, 768)，已 L2 归一化
    eval_text_bank = text_bank.detach().cpu()             # 评估统一在 CPU 上算
    print(
        f"[Text] text_bank={tuple(text_bank.shape)} dtype={text_bank.dtype} "
        f"({time.time() - text_started:.1f}s, 全量库负样本)"
    )

    model_kwargs = dict(
        num_classes=num_classes,
        E_in=n_channels,
        E_out=args.e_out,
        C0=args.c0,
        stem_k=args.stem_k,
        dropout=args.dropout,
        use_electrode_attn=args.use_electrode_attn,
        embed_dim=text_encoder.embedding_dim,
        proj_hidden=args.proj_hidden,
    )
    model = PMCNN(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] PMCNN-alignment | embed_dim={text_encoder.embedding_dim} "
        f"| E_in={n_channels} E_out={args.e_out} | params={n_params:,}"
    )
    print(
        f"[Objective] InfoNCE(全量文本库, tau={args.temperature:g})"
        + (f" + {args.ce_weight:g} * CE" if args.ce_weight > 0 else "")
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

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 3)
    )

    run_name = args.run_name or f"align_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    print(f"[Output] {out_dir}")

    split_args = {
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.split_seed,
    }

    best_score, best_epoch, no_improve = -1.0, 0, 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_align, train_cls, train_n = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            text_bank,
            args.temperature,
            args.ce_weight,
            args.label_smoothing,
            args.max_train_batches,
            amp_dtype,
            scaler,
        )

        # 每轮只在验证集内部检索，开销很小（246 × 768 点积）
        z_val, idx_val, _ = extract_embeddings(
            model, val_loader, device, args.max_val_batches, amp_dtype
        )
        val_metrics = retrieval_metrics(
            z_val, eval_text_bank[idx_val], torch.arange(len(idx_val)), args.temperature
        )
        val_full = retrieval_metrics(
            z_val, eval_text_bank, idx_val, args.temperature
        )
        score = val_metrics[args.select_metric]
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_align": train_align,
            "train_cls": train_cls,
            "train_samples": train_n,
            "val_loss": val_metrics["loss"],
            "val_r1": val_metrics["r1"],
            "val_r5": val_metrics["r5"],
            "val_r10": val_metrics["r10"],
            "val_mrr": val_metrics["mrr"],
            "val_median_rank": val_metrics["median_rank"],
            "val_full_r1": val_full["r1"],
            "val_full_mrr": val_full["mrr"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv", index=False, encoding="utf-8-sig"
        )

        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"align={train_align:.4f} "
            + (f"cls={train_cls:.4f} " if args.ce_weight > 0 else "")
            + f"val_R@1={val_metrics['r1']:.4f} val_MRR={val_metrics['mrr']:.4f} "
            f"val_med={val_metrics['median_rank']:.1f} "
            f"full_R@1={val_full['r1']:.4f} "
            f"lr={row['lr']:.2e} ({row['seconds']:.1f}s)"
        )

        if score > best_score + 1e-4:
            best_score, best_epoch, no_improve = score, epoch, 0
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
                args.text_model,
                text_encoder.embedding_dim,
                args.temperature,
                args.select_metric,
                epoch,
                score,
                val_metrics,
            )
            print(f"[Best] epoch={epoch} {args.select_metric}={best_score:.4f} -> {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[Stop] 验证集 {args.select_metric} 已 {args.patience} 个 epoch 未提升，提前结束")
                break

    print(f"[Done] best epoch={best_epoch} {args.select_metric}={best_score:.4f}")

    # ---------- 用 best.pt 做最终报告 ----------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # 训练集 EEG 向量作为 kNN 诊断的 gallery（只算一次）
    z_train, _, y_train = extract_embeddings(
        model, train_loader, device, args.max_train_batches, amp_dtype
    )
    train_gallery = (z_train, y_train)

    report_split(
        model, val_loader, eval_text_bank, out_dir, "val",
        args.temperature, device, num_classes,
        args.max_val_batches, amp_dtype, train_gallery, args.knn_k,
    )

    if not args.skip_test:
        report_split(
            model, test_loader, eval_text_bank, out_dir, "test",
            args.temperature, device, num_classes,
            args.max_test_batches, amp_dtype, train_gallery, args.knn_k,
        )


if __name__ == "__main__":
    main()
