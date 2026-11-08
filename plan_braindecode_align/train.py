"""EEG-文本嵌入空间对齐（top-10 类，braindecode 标准模型 + PMCNN + 冻结 LaBSE）。

把 ``plan_braindecode_baseline`` 的分类对照改造成跨模态对齐实验：

- 目标从 10 类交叉熵改为 InfoNCE：EEG 经主干 + 投影头得到 ``embed_dim`` 维
  L2 归一化向量，与冻结 LaBSE 的句向量对齐；
- 负样本使用**全量文本库**（数据集里每个试次对应的句子），而不是 batch 内负样本；
- 文本向量在训练开始前一次性算好，训练循环里只做查表；``--model all`` 时
  这份文本库也只算一次，所有候选模型共用；
- 评估改为跨模态检索（R@1/5/10、MRR、median rank），并用 kNN 分类做语义结构
  诊断；早停默认看验证集 MRR；
- 分类头保留但默认不参与训练（``--ce-weight`` 默认 0，纯对齐）。

数据、64 通道子集、z-score、重采样到 250 Hz、随机分层 8:1:1 划分全部沿用
``plan_braindecode_baseline``，保证两套实验在同一划分上可比。

用法::

    # 单个模型
    uv run plan_braindecode_align/train.py --model eegnet

    # 全部模型依次训练（文本库只编码一次）
    uv run plan_braindecode_align/train.py --model all

    # 快速试跑
    uv run plan_braindecode_align/train.py --model all --epochs 2 \
        --max-train-batches 5 --max-val-batches 2 --skip-test
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from dataset import TOP_ROOT, TopClassEEGDataset, split_indices
from models import available_models, build_align_model
from text_encoder import LaBSETextEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "braindecode_align"


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
        text_bank: ``(N, D)``，全量句子的冻结 LaBSE 向量。
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

        # 对比损失用 fp32 算，避免 bf16 下高维点积的精度损失
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
        "preds": preds,
        "targets": targets,
    }


@torch.no_grad()
def retrieval_top1_labels(z_query, bank, bank_labels):
    """检索 top-1 候选文本所属的类别，作为“检索口径”的预测标签。

    这是跨模态检索任务真正的预测方式：EEG 向量在文本库里找最相似的句子，
    该句子的类别就是预测类别。与 kNN 口径相比，它衡量的是检索错误结构。
    """
    z = F.normalize(z_query.detach().float().cpu(), dim=-1)
    bank = F.normalize(bank.detach().float().cpu(), dim=-1)
    top1 = (z @ bank.t()).argmax(dim=1)
    return bank_labels[top1.numpy()]


def plot_confusion_heatmap(csv_path: Path, out_path: Path, title: str) -> None:
    """读取 ``*_confusion_matrix.csv``，用 seaborn 画一张热力图。"""
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
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"[Plot] 混淆矩阵热力图已保存: {out_path}")


def save_confusion(
    preds, targets, num_classes, class_names, out_dir, stem, title
) -> Path:
    """落 ``{stem}_confusion_matrix.csv`` 并画对应热力图，返回 csv 路径。"""
    names = [class_names[index] for index in range(num_classes)]
    matrix = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    csv_path = out_dir / f"{stem}_confusion_matrix.csv"
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        csv_path, encoding="utf-8-sig"
    )
    plot_confusion_heatmap(csv_path, out_dir / f"{stem}_confusion_matrix.png", title)
    return csv_path


def save_checkpoint(
    path,
    model,
    model_name,
    model_kwargs,
    config,
    epoch,
    score,
    metrics,
):
    """保存模型与复现实验所需的全部配置。"""
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "config": config,
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
    class_names=None,
    bank_labels=None,
):
    """在指定划分上算检索指标（本划分内部 + 全量文本库）与 kNN 诊断。

    若给了 ``class_names`` / ``bank_labels``，额外落两种口径的混淆矩阵：

    - ``{split}_knn_confusion_matrix.*``：kNN 分类器的预测；
    - ``{split}_retrieval_confusion_matrix.*``：top-1 检索到的文本所属类别。
    """
    z_eeg, indices, labels = extract_embeddings(
        model, loader, device, max_batches, amp_dtype
    )

    # 1) 只在当前划分内部检索：难度较低，适合看训练是否有效
    in_split = retrieval_metrics(
        z_eeg, text_bank[indices], torch.arange(len(indices)), temperature
    )
    # 2) 在全量文本库中检索：更接近真实使用场景
    full_bank = retrieval_metrics(z_eeg, text_bank, indices, temperature)

    report = {
        "split": split_name,
        "retrieval_in_split": in_split,
        "retrieval_full_bank": full_bank,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    can_confuse = class_names is not None and len(class_names) == num_classes

    if train_gallery is not None:
        z_train, y_train = train_gallery
        knn = knn_diagnostic(
            z_train, y_train, z_eeg, labels, k=knn_k, num_classes=num_classes
        )
        knn_preds, knn_targets = knn.pop("preds"), knn.pop("targets")
        report["knn"] = knn
        if can_confuse:
            cm_path = save_confusion(
                knn_preds, knn_targets, num_classes, class_names, out_dir,
                f"{split_name}_knn", f"kNN 混淆矩阵（{split_name}，{num_classes} 类）"
            )
            report["knn"]["confusion_matrix"] = cm_path.name

    if bank_labels is not None and can_confuse:
        retrieval_preds = retrieval_top1_labels(z_eeg, text_bank, bank_labels)
        cm_path = save_confusion(
            retrieval_preds, bank_labels[indices.numpy()], num_classes,
            class_names, out_dir, f"{split_name}_retrieval",
            f"检索混淆矩阵（{split_name}，{num_classes} 类）"
        )
        report["retrieval_confusion_matrix"] = cm_path.name

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


def run_experiment(
    args,
    dataset,
    split,
    model_name,
    run_dir,
    device,
    amp_dtype,
    scaler,
    text_bank,
    eval_text_bank,
    embed_dim,
):
    """训练并评估单个模型，返回一行汇总结果。"""
    set_seed(args.seed)  # 每个模型都从同一随机状态出发，batch 顺序一致

    train_idx, val_idx, test_idx = split
    train_loader, val_loader, test_loader = make_loaders(
        dataset, train_idx, val_idx, test_idx, args.batch_size, device.type == "cuda"
    )

    n_chans = dataset.n_channels
    n_times = int(dataset.data.shape[-1])
    num_classes = len(dataset.class_ids)

    model_kwargs = dict(
        name=model_name,
        n_chans=n_chans,
        n_times=n_times,
        num_classes=num_classes,
        sfreq=args.target_sfreq,
        embed_dim=embed_dim,
        proj_hidden=args.proj_hidden,
        drop_prob=args.drop_prob,
        use_electrode_attn=args.use_electrode_attn,
        e_out=args.e_out,
        c0=args.c0,
        stem_k=args.stem_k,
    )
    model = build_align_model(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[Model] {model_name} | params={n_params:,} | feature_dim={model.feature_dim} "
        f"| E_in={n_chans} T={n_times} classes={num_classes}"
    )

    config = {
        "model": model_name,
        "subject": args.subject,
        "n_params": n_params,
        "feature_dim": model.feature_dim,
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": num_classes,
        "sfreq": args.target_sfreq,
        "normalize": not args.no_normalize,
        "use_64_channels": args.use_64_channels,
        "split": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.split_seed,
        },
        "text_model": args.text_model,
        "embed_dim": embed_dim,
        "temperature": args.temperature,
        "proj_hidden": args.proj_hidden,
        "ce_weight": args.ce_weight,
        "select_metric": args.select_metric,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "drop_prob": args.drop_prob,
        "use_electrode_attn": args.use_electrode_attn,
        "class_ids": dataset.class_ids,
        "class_names": dataset.class_names,
    }

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.lr_patience)
    )

    checkpoint_path = run_dir / "best.pt"
    best_score, best_epoch, no_improve = -1.0, 0, 0
    history: list[dict] = []
    started_all = time.time()

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

        # 每轮只在验证集内部检索，开销很小（数百 × embed_dim 点积）
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

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(score)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            # 刚衰减：模型要几个 epoch 才适应新步长，早停计数重新起算
            no_improve = 0
            print(f"[LR] {lr_before:.2e} -> {lr_after:.2e}，早停计数重置")

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
            "lr": lr_after,
            "seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            run_dir / "history.csv", index=False, encoding="utf-8-sig"
        )

        print(
            f"[{model_name} | Epoch {epoch:02d}/{args.epochs}] "
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
                model_name,
                model_kwargs,
                config,
                epoch,
                score,
                val_metrics,
            )
            print(
                f"[Best] epoch={epoch} {args.select_metric}={best_score:.4f} "
                f"-> {checkpoint_path}"
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    f"[Stop] 验证集 {args.select_metric} 已 {args.patience} "
                    f"个 epoch 未提升，提前结束"
                )
                break

    print(f"[Done] {model_name} best epoch={best_epoch} {args.select_metric}={best_score:.4f}")

    result = {
        "model": model_name,
        "params": n_params,
        "feature_dim": model.feature_dim,
        "embed_dim": embed_dim,
        "best_epoch": best_epoch,
        "val_select_metric": args.select_metric,
        "val_select": best_score,
        "test_r1": None,
        "test_r5": None,
        "test_r10": None,
        "test_mrr": None,
        "test_median_rank": None,
        "test_knn_acc": None,
        "test_knn_macro_f1": None,
        "train_seconds": round(time.time() - started_all, 1),
    }

    if args.skip_test:
        print(f"[Skip] {model_name}: --skip-test，不评估测试集")
        return result

    # ---------- 用 best.pt 做最终报告 ----------
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # 训练集 EEG 向量作为 kNN 诊断的 gallery（只算一次）
    z_train, _, y_train = extract_embeddings(
        model, train_loader, device, args.max_train_batches, amp_dtype
    )
    train_gallery = (z_train, y_train)

    report_split(
        model, val_loader, eval_text_bank, run_dir, "val",
        args.temperature, device, num_classes,
        args.max_val_batches, amp_dtype, train_gallery, args.knn_k,
        dataset.class_names, dataset.labels,
    )
    test_report = report_split(
        model, test_loader, eval_text_bank, run_dir, "test",
        args.temperature, device, num_classes,
        args.max_test_batches, amp_dtype, train_gallery, args.knn_k,
        dataset.class_names, dataset.labels,
    )

    test_full = test_report["retrieval_full_bank"]
    result["test_r1"] = test_full["r1"]
    result["test_r5"] = test_full["r5"]
    result["test_r10"] = test_full["r10"]
    result["test_mrr"] = test_full["mrr"]
    result["test_median_rank"] = test_full["median_rank"]
    result["test_knn_acc"] = test_report["knn"]["accuracy"]
    result["test_knn_macro_f1"] = test_report["knn"]["macro_f1"]
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG-文本嵌入对齐（top-10 类，braindecode 标准模型 + PMCNN + 冻结 LaBSE）。"
    )
    # ---------- 数据 ----------
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--split-seed", type=int, default=42, help="随机划分种子（各模型共用）")
    parser.add_argument("--target-sfreq", type=float, default=250.0, help="重采样目标采样率（braindecode/MNE 抗混叠重采样，默认 250 Hz）")
    parser.add_argument("--no-normalize", action="store_true", help="关闭逐通道 z-score")
    parser.add_argument("--use-64-channels", action=argparse.BooleanOptionalAction, default=True, help="是否只用 channels_64 的 64 个通道（默认开启）")
    # ---------- 模型 ----------
    parser.add_argument("--model", default="eegnet", choices=[*available_models(), "all"], help="要训练的模型，all 表示全部依次训练")
    parser.add_argument("--drop-prob", type=float, default=0.5, help="模型内部 dropout 概率")
    parser.add_argument("--e-out", type=int, default=16, help="PMCNN Stem 压缩后的电极维 E_out")
    parser.add_argument("--c0", type=int, default=64, help="PMCNN Stem 输出通道数")
    parser.add_argument("--stem-k", type=int, default=7, help="PMCNN Stem 时间卷积核长")
    parser.add_argument("--use-electrode-attn", action=argparse.BooleanOptionalAction, default=True, help="PMCNN 是否在电极轴上做注意力池化（默认开启）")
    # ---------- 对齐 ----------
    parser.add_argument("--text-model", default="sentence-transformers/LaBSE", help="LaBSE 模型名或本地路径")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE 温度 tau")
    parser.add_argument("--proj-hidden", type=int, default=512, help="投影头隐层维度")
    parser.add_argument("--ce-weight", type=float, default=0.0, help="辅助分类 CE 权重（0 表示纯对齐）")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="辅助 CE 的 label smoothing")
    # ---------- 训练 ----------
    parser.add_argument("--epochs", type=int, default=50, help="最大 epoch 数")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度（默认开启）")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--patience", type=int, default=10, help="验证指标无提升的容忍 epoch 数（每个学习率阶段独立计算）")
    parser.add_argument("--lr-patience", type=int, default=3, help="学习率衰减前，验证指标无提升的容忍 epoch 数")
    parser.add_argument("--select-metric", default="mrr", choices=("mrr", "r1", "r5"), help="早停与保存 best 使用的验证指标")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
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

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
    print(
        f"[Channels] {dataset.n_channels} "
        f"({'channels_64 顺序' if args.use_64_channels else 'fif 原顺序'}) | "
        f"T={dataset.data.shape[-1]} | sfreq={args.target_sfreq:g} Hz"
    )

    if not all(text.strip() for text in dataset.texts):
        raise RuntimeError("数据中有空句子，无法做文本对齐")

    split = split_indices(dataset.labels, args.val_ratio, args.test_ratio, args.split_seed)
    for name, idx in zip(("train", "val", "test"), split):
        counts = np.bincount(dataset.labels[idx], minlength=num_classes)
        print(f"[Split] {name:<5} {len(idx):>5} 条 | 每类 {counts.min()}-{counts.max()}")

    # ---------- 预计算全量文本向量库（所有模型共用一份） ----------
    print(f"[Text] 正在用 {args.text_model} 编码 {len(dataset.texts)} 个句子 ...", flush=True)
    text_started = time.time()
    text_encoder = LaBSETextEncoder(args.text_model)
    text_bank = text_encoder(dataset.texts).to(device)   # (N, embed_dim)，已 L2 归一化
    eval_text_bank = text_bank.detach().cpu()             # 评估统一在 CPU 上算
    embed_dim = text_encoder.embedding_dim
    print(
        f"[Text] text_bank={tuple(text_bank.shape)} dtype={text_bank.dtype} "
        f"({time.time() - text_started:.1f}s, 全量库负样本)"
    )

    # 混合精度：Ampere 及更新架构用 bf16（无需 GradScaler），否则退回 fp16 + GradScaler
    amp_dtype, scaler = None, None
    if device.type == "cuda" and not args.no_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    print(f"[Precision] {amp_dtype or torch.float32} | batch={args.batch_size}")

    model_names = available_models() if args.model == "all" else [args.model]
    run_name = args.run_name or f"align_{datetime.now():%Y%m%d_%H%M%S}"
    run_root = args.out_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[Output] {run_root}\n[Models] {', '.join(model_names)}")
    print(
        f"[Objective] InfoNCE(全量文本库, tau={args.temperature:g}, embed_dim={embed_dim})"
        + (f" + {args.ce_weight:g} * CE" if args.ce_weight > 0 else "")
    )

    results: list[dict] = []
    for model_name in model_names:
        model_dir = run_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        results.append(
            run_experiment(
                args,
                dataset,
                split,
                model_name,
                model_dir,
                device,
                amp_dtype,
                scaler,
                text_bank,
                eval_text_bank,
                embed_dim,
            )
        )
        # 每个模型跑完就落一次汇总，中途失败也不至于全丢
        pd.DataFrame(results).to_csv(
            run_root / "summary.csv", index=False, encoding="utf-8-sig"
        )

    summary = pd.DataFrame(results)
    summary_path = run_root / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 108)
    print("EEG-文本对齐汇总（检索指标为测试集全量文本库口径）")
    print("=" * 108)
    print(
        f"{'model':<15}{'params':>10}{'feat':>7}{'best_ep':>9}"
        f"{'val_' + args.select_metric:>12}{'test_R@1':>10}{'test_R@5':>10}"
        f"{'test_R@10':>11}{'test_MRR':>10}{'knn_acc':>10}{'seconds':>9}"
    )

    def fmt(value) -> str:
        """None（未跑测试集）显示为破折号，避免与 0 混淆。"""
        return "—" if value is None else f"{value:.4f}"

    for row in results:
        print(
            f"{row['model']:<15}{row['params']:>10,}{row['feature_dim']:>7}"
            f"{row['best_epoch']:>9}{row['val_select']:>12.4f}"
            f"{fmt(row['test_r1']):>10}{fmt(row['test_r5']):>10}"
            f"{fmt(row['test_r10']):>11}{fmt(row['test_mrr']):>10}"
            f"{fmt(row['test_knn_acc']):>10}{row['train_seconds']:>9.1f}"
        )
    print("=" * 108)
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
