"""EEGNet 时域基线的评估脚本。

本目录与 ``experiments/deepnet/freqcnn`` 完全解耦，不共享模型或训练代码。
支持五种与线性基线对齐的评估协议：

1. ``holdout``：按句类分层的 8:2 随机划分（与 baseline_lda 的 test 口径一致）；
2. ``cv5``：按句类分层的随机 5 折（同 run 的试次会同时进入训练与测试）；
3. ``loro``：留一 run（leave-one-run-out）；
4. ``lodo``：留一采集日（leave-one-day-out）；
5. ``runid``：把标签换成 run 身份，做随机 5 折负对照（45/44 类）。

不指定 ``--protocol`` 时默认依次运行 ``cv5``、``loro``、``lodo``；``--protocol all``
表示五种全部运行。

每个协议都会输出池化准确率（附 run 级 cluster bootstrap 95% CI）、逐折均值与
标准差、宏平均 F1、多数类基线，以及可选的置换零分布。深度模型的置换检验需要
对每个置换重新训练全部折，成本很高，默认关闭。

用法::

    # 随机 5 折
    uv run experiments/deepnet/eegnet/train.py --subject sub-01 --protocol cv5

    # 5 名被试，三种协议
    uv run experiments/deepnet/eegnet/train.py \
        --subject sub-01 sub-02 sub-03 sub-04 sub-05 \
        --protocol cv5 loro lodo --run-name main_8_2

    # 冒烟：2 类、2 折、1 epoch
    uv run experiments/deepnet/eegnet/train.py \
        --subject sub-01 --top-k 2 --folds 2 --epochs 1 --protocol cv5

    # 降采样到 250 Hz（带抗混叠重采样，时间核按秒自动缩放）
    uv run experiments/deepnet/eegnet/train.py \
        --subject sub-01 --target-sfreq 250 --protocol cv5

注意：

- 深度模型的训练轮数是固定的，不做基于测试集的早停，避免把测试信息带回训练；
- LORO 有 45 折，成本远高于 cv5；建议先用单被试跑通，再扩展到 5 人；
- 若同时报告多个被试、多个协议或多个配置的 p 值，需要在论文中说明多重比较校正。
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
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from dataset import CSV_ROOT, PKL_ROOT, EEGNetDataset, split_indices
from model import build_model, count_parameters, kernel_length_samples

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "deepnet" / "eegnet"
MODEL_NAME = "EEGNet"

REPORT_FIELDS = ["subject", "protocol", "label", "precision", "recall", "f1", "support"]
FOLD_FIELDS = [
    "subject",
    "protocol",
    "split",
    "n_train",
    "n_test",
    "accuracy",
    "macro_f1",
    "majority_baseline_accuracy",
    "train_loss",
    "train_accuracy",
]
SUMMARY_FIELDS = [
    "subject",
    "eeg_task",
    "protocol",
    "model",
    "target_sfreq",
    "n_channels",
    "n_times",
    "n_classes",
    "n_samples",
    "n_splits",
    "n_params",
    "epochs",
    "batch_size",
    "lr",
    "seed",
    "chance",
    "accuracy",
    "ci_low",
    "ci_high",
    "macro_f1",
    "weighted_f1",
    "fold_mean_accuracy",
    "fold_std_accuracy",
    "fold_mean_macro_f1",
    "fold_std_macro_f1",
    "majority_baseline_accuracy",
    "permutation_n",
    "permutation_null_mean",
    "permutation_null_p95",
    "permutation_z",
    "permutation_p",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，并关闭 cudnn 非确定性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但当前环境没有可用 GPU")
    return device


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """把字典列表写成 UTF-8-SIG CSV（Excel 可直接打开）。"""
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_classification_csv(
    path: Path, subject: str, protocol: str, report: dict
) -> None:
    """把池化 ``classification_report(output_dict=True)`` 写成逐类 CSV。"""
    total_support = sum(
        metrics["support"]
        for label, metrics in report.items()
        if isinstance(metrics, dict)
        and label not in ("macro avg", "weighted avg", "samples avg")
    )
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {
                "subject": subject,
                "protocol": protocol,
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
        elif label == "accuracy":  # output_dict 模式下 accuracy 是标量
            row = {
                "subject": subject,
                "protocol": protocol,
                "label": label,
                "precision": metrics,
                "recall": metrics,
                "f1": metrics,
                "support": total_support,
            }
        else:
            continue
        rows.append(row)
    write_rows(path, rows, REPORT_FIELDS)


def bootstrap_accuracy_ci(
    y_true, y_pred, groups, seed: int, n_boot: int = 5000
) -> tuple[float, float]:
    """按 run 重采样的准确率 95% CI（cluster bootstrap）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([index_by_group[j] for j in pick])
        accs[i] = np.mean(y_true[index] == y_pred[index])
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


class LabeledSubset(Dataset):
    """按给定下标取 base 的样本，并用外部标签覆盖 base 的标签。

    用于把同一份数据在 ``runid`` 协议下换成 run 标签，而不修改 base 数据集。
    """

    def __init__(self, base: Dataset, indices: np.ndarray, labels: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        base_index = int(self.indices[index])
        x, _ = self.base[base_index]
        return x, int(self.labels[base_index])


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def build_splits(
    protocol: str,
    labels: np.ndarray,
    run_groups: np.ndarray,
    day_groups: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], np.ndarray]:
    """返回 ``(splits, ci_groups)``，每个 split 为 ``(name, train_idx, test_idx)``。"""
    labels = np.asarray(labels)
    if protocol == "holdout":
        train_idx, _, test_idx = split_indices(labels, 0.0, 0.2, seed)
        return [("holdout", train_idx, test_idx)], run_groups
    if protocol in ("cv5", "runid"):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = [
            (f"fold_{index}", train_idx, test_idx)
            for index, (train_idx, test_idx) in enumerate(
                splitter.split(np.zeros(len(labels)), labels), start=1
            )
        ]
        return splits, run_groups
    if protocol == "loro":
        splits = [
            (f"run_{int(run_groups[test_idx][0]):02d}", train_idx, test_idx)
            for train_idx, test_idx in LeaveOneGroupOut().split(
                np.zeros(len(labels)), labels, run_groups
            )
        ]
        return splits, run_groups
    if protocol == "lodo":
        splits = [
            (f"day_{int(day_groups[test_idx][0]):02d}", train_idx, test_idx)
            for train_idx, test_idx in LeaveOneGroupOut().split(
                np.zeros(len(labels)), labels, day_groups
            )
        ]
        return splits, day_groups
    raise ValueError(f"未知协议: {protocol}")


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_fold(
    args,
    dataset: EEGNetDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    task_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    fold_seed: int,
    log_prefix: str = "",
) -> tuple[np.ndarray, float, float]:
    """训练一个划分并返回 ``(test_pred, final_train_loss, final_train_acc)``。"""
    set_seed(fold_seed)
    model = build_model(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        n_classes=num_classes,
        sfreq=dataset.sfreq,
        f1=args.f1,
        d=args.d,
        f2=args.f2,
        kernel_seconds=args.kernel_seconds,
        separable_seconds=args.separable_seconds,
        dropout=args.dropout,
        dropout2=args.dropout2,
    ).to(device)

    train_set = LabeledSubset(dataset, train_idx, task_labels)
    test_set = LabeledSubset(dataset, test_idx, task_labels)
    generator = torch.Generator()
    generator.manual_seed(fold_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    criterion = nn.CrossEntropyLoss()
    prefix = f"{log_prefix} " if log_prefix else ""
    total_steps = len(train_loader)
    print(
        f"{prefix}start | train={len(train_idx)} test={len(test_idx)} "
        f"epochs={args.epochs} steps/epoch={total_steps}",
        flush=True,
    )

    last_loss = 0.0
    last_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.perf_counter()
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_pred = logits.argmax(dim=1)
            batch_correct = int((batch_pred == y).sum().item())
            batch_loss = float(loss.item())
            total_loss += batch_loss * len(y)
            correct += batch_correct
            total += len(y)
            print(
                f"{prefix}epoch {epoch + 1:03d}/{args.epochs} "
                f"step {step:03d}/{total_steps:03d} "
                f"loss={batch_loss:.4f} acc={batch_correct / max(len(y), 1):.4f} "
                f"lr={float(optimizer.param_groups[0]['lr']):.2e}",
                flush=True,
            )
        scheduler.step()
        last_loss = total_loss / max(total, 1)
        last_acc = correct / max(total, 1)
        print(
            f"{prefix}epoch {epoch + 1:03d}/{args.epochs} done | "
            f"train_loss={last_loss:.4f} train_acc={last_acc:.4f} | "
            f"lr={float(optimizer.param_groups[0]['lr']):.2e} | "
            f"elapsed={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device, non_blocking=True)
            predictions.append(model(x).argmax(dim=1).cpu().numpy())
    pred = (
        np.concatenate(predictions).astype(np.int64)
        if predictions
        else np.empty(0, dtype=np.int64)
    )
    return pred, last_loss, last_acc


def evaluate_protocol(
    args,
    protocol: str,
    dataset: EEGNetDataset,
    task_labels: np.ndarray,
    class_names: list[str],
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    ci_groups: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> dict:
    """在给定划分上训练/预测，返回池化指标与逐折明细。"""
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    fold_rows: list[dict] = []

    for fold_index, (name, train_idx, test_idx) in enumerate(splits, start=1):
        print(
            f"    [{protocol}] {name}: train={len(train_idx)} test={len(test_idx)}",
            flush=True,
        )
        pred, train_loss, train_acc = train_one_fold(
            args,
            dataset,
            train_idx,
            test_idx,
            task_labels,
            num_classes,
            device,
            fold_seed=args.seed + fold_index,
            log_prefix=f"[{args.subject}][{protocol}][{name}]",
        )
        y_true = task_labels[test_idx]
        majority = int(
            np.bincount(task_labels[train_idx], minlength=num_classes).argmax()
        )
        fold_rows.append(
            {
                "subject": args.subject,
                "protocol": protocol,
                "split": name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_true, pred)),
                "macro_f1": float(
                    f1_score(y_true, pred, average="macro", zero_division=0)
                ),
                "majority_baseline_accuracy": float(np.mean(y_true == majority)),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
            }
        )
        y_true_parts.append(y_true)
        y_pred_parts.append(pred)
        group_parts.append(ci_groups[test_idx])

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    pooled_groups = np.concatenate(group_parts)
    ci_low, ci_high = bootstrap_accuracy_ci(
        y_true, y_pred, pooled_groups, args.seed, args.bootstrap
    )

    accs = np.asarray([row["accuracy"] for row in fold_rows])
    f1s = np.asarray([row["macro_f1"] for row in fold_rows])
    majority_correct = sum(
        row["majority_baseline_accuracy"] * row["n_test"] for row in fold_rows
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "fold_mean_accuracy": float(accs.mean()),
        "fold_std_accuracy": float(accs.std()),
        "fold_mean_macro_f1": float(f1s.mean()),
        "fold_std_macro_f1": float(f1s.std()),
        "majority_baseline_accuracy": float(majority_correct / len(y_true)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
        "per_class_recall": [
            float(np.mean(y_pred[y_true == c] == c)) if np.any(y_true == c) else None
            for c in range(num_classes)
        ],
        "n_samples": int(len(y_true)),
        "n_splits": int(len(fold_rows)),
        "folds": fold_rows,
    }


def permutation_null(
    args,
    protocol: str,
    dataset: EEGNetDataset,
    task_labels: np.ndarray,
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """全局打乱标签后重新训练全部折，返回池化准确率零分布。

    成本约为 ``permutations × 折数 × epochs``，默认不开启。
    """
    rng = np.random.default_rng(args.seed + 10_000)
    null = np.empty(args.permutations, dtype=np.float64)
    for perm_index in range(args.permutations):
        shuffled = rng.permutation(task_labels)
        correct = 0
        for fold_index, (_, train_idx, test_idx) in enumerate(splits, start=1):
            pred, _, _ = train_one_fold(
                args,
                dataset,
                train_idx,
                test_idx,
                shuffled,
                num_classes,
                device,
                fold_seed=args.seed + fold_index,
                log_prefix=f"[{args.subject}][{protocol}][perm {perm_index + 1}]",
            )
            correct += int(np.sum(pred == shuffled[test_idx]))
        null[perm_index] = correct / len(shuffled)
        print(
            f"    [置换] {perm_index + 1}/{args.permutations} "
            f"acc={null[perm_index]:.4f}",
            flush=True,
        )
    return null


# ---------------------------------------------------------------------------
# 单被试入口
# ---------------------------------------------------------------------------
def run_subject(
    args, dataset: EEGNetDataset, out_dir: Path, device: torch.device
) -> list[dict]:
    y = dataset.labels
    class_names = list(dataset.class_names)
    run_groups = dataset.run_groups
    day_groups = dataset.day_groups

    run_ids = sorted({int(run) for run in run_groups})
    run_to_label = {run: index for index, run in enumerate(run_ids)}
    y_run = np.asarray(
        [run_to_label[int(run)] for run in run_groups], dtype=np.int64
    )
    run_names = [f"run_{run:02d}" for run in run_ids]

    # 参数量按主任务（句类）统计；runid 只改变分类头大小
    n_params = count_parameters(
        build_model(
            n_channels=dataset.n_channels,
            n_times=dataset.n_times,
            n_classes=len(class_names),
            sfreq=dataset.sfreq,
            f1=args.f1,
            d=args.d,
            f2=args.f2,
            kernel_seconds=args.kernel_seconds,
            separable_seconds=args.separable_seconds,
            dropout=args.dropout,
            dropout2=args.dropout2,
        )
    )

    if "all" in args.protocol:
        protocols = ["holdout", "cv5", "loro", "lodo", "runid"]
    else:
        protocols = list(dict.fromkeys(args.protocol))

    details: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for protocol in protocols:
        if protocol == "runid":
            task_labels = y_run
            task_names = run_names
            num_classes = len(run_ids)
            min_run_count = int(np.bincount(y_run).min())
            if min_run_count < 2:
                raise ValueError(
                    "runid 协议要求每个 run 至少有 2 条样本；"
                    "请使用完整的 top-k 数据，或从 --protocol 中去掉 runid"
                )
            protocol_folds = min(args.folds, min_run_count)
        else:
            task_labels = y
            task_names = class_names
            num_classes = len(class_names)
            min_class_count = int(np.bincount(y).min())
            if min_class_count < 2:
                raise ValueError(
                    f"句类 {protocol} 协议要求每类至少 2 条样本，当前最小 {min_class_count}"
                )
            protocol_folds = min(args.folds, min_class_count)

        splits, ci_groups = build_splits(
            protocol,
            task_labels,
            run_groups,
            day_groups,
            protocol_folds,
            args.seed,
        )
        print(
            f"\n----- [{protocol}] {len(splits)} 折，{num_classes} 类，"
            f"chance={1 / num_classes:.4f} -----",
            flush=True,
        )
        result = evaluate_protocol(
            args,
            protocol,
            dataset,
            task_labels,
            task_names,
            splits,
            ci_groups,
            num_classes,
            device,
        )
        print(
            f"[{protocol}] 池化 acc={result['accuracy']:.4f} "
            f"95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
            f"macro_f1={result['macro_f1']:.4f} | 逐折 "
            f"{result['fold_mean_accuracy']:.4f} ± {result['fold_std_accuracy']:.4f} "
            f"| 多数类={result['majority_baseline_accuracy']:.4f} "
            f"chance={1 / num_classes:.4f}",
            flush=True,
        )

        report = classification_report(
            result["y_true"],
            result["y_pred"],
            labels=list(range(num_classes)),
            target_names=task_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        save_classification_csv(
            out_dir / f"classification_report_{protocol}_{args.subject}.csv",
            args.subject,
            protocol,
            report,
        )
        write_rows(
            out_dir / f"folds_{protocol}_{args.subject}.csv",
            result["folds"],
            FOLD_FIELDS,
        )

        permutation = None
        if args.permutations > 0:
            null = permutation_null(
                args, protocol, dataset, task_labels, splits, num_classes, device
            )
            observed = float(result["accuracy"])
            permutation = {
                "n": args.permutations,
                "null_mean": float(null.mean()),
                "null_std": float(null.std()),
                "null_p95": float(np.percentile(null, 95)),
                "observed": observed,
                "z_score": float((observed - null.mean()) / (null.std() + 1e-12)),
                "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
                "null_samples": null.tolist(),
            }

        perm = permutation or {}
        summary_rows.append(
            {
                "subject": args.subject,
                "eeg_task": args.task,
                "protocol": protocol,
                "model": MODEL_NAME,
                "target_sfreq": dataset.sfreq,
                "n_channels": dataset.n_channels,
                "n_times": dataset.n_times,
                "n_classes": num_classes,
                "n_samples": result["n_samples"],
                "n_splits": result["n_splits"],
                "n_params": n_params,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "chance": float(1 / num_classes),
                "accuracy": result["accuracy"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "fold_mean_accuracy": result["fold_mean_accuracy"],
                "fold_std_accuracy": result["fold_std_accuracy"],
                "fold_mean_macro_f1": result["fold_mean_macro_f1"],
                "fold_std_macro_f1": result["fold_std_macro_f1"],
                "majority_baseline_accuracy": result["majority_baseline_accuracy"],
                "permutation_n": perm.get("n"),
                "permutation_null_mean": perm.get("null_mean"),
                "permutation_null_p95": perm.get("null_p95"),
                "permutation_z": perm.get("z_score"),
                "permutation_p": perm.get("p_value"),
            }
        )

        details[protocol] = {
            "class_names": task_names,
            "accuracy": result["accuracy"],
            "ci": [result["ci_low"], result["ci_high"]],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "fold_mean_accuracy": result["fold_mean_accuracy"],
            "fold_std_accuracy": result["fold_std_accuracy"],
            "fold_mean_macro_f1": result["fold_mean_macro_f1"],
            "fold_std_macro_f1": result["fold_std_macro_f1"],
            "majority_baseline_accuracy": result["majority_baseline_accuracy"],
            "n_samples": result["n_samples"],
            "n_splits": result["n_splits"],
            "confusion_matrix": result["confusion_matrix"],
            "per_class_recall": result["per_class_recall"],
            "folds": result["folds"],
            "permutation": permutation,
        }

    payload = {
        "subject": args.subject,
        "task": "eegnet_deepnet_baseline",
        "eeg_task": args.task,
        "model": MODEL_NAME,
        "dataset": {
            "n_samples": int(len(y)),
            "n_channels": int(dataset.n_channels),
            "n_times": int(dataset.n_times),
            "sfreq": float(dataset.sfreq),
            "n_runs": int(len(run_ids)),
        },
        "model_config": {
            "f1": args.f1,
            "d": args.d,
            "f2": args.f2,
            "kernel_seconds": args.kernel_seconds,
            "separable_seconds": args.separable_seconds,
            "kernel_length_samples": kernel_length_samples(
                args.kernel_seconds, dataset.sfreq
            ),
            "separable_kernel_samples": kernel_length_samples(
                args.separable_seconds, dataset.sfreq
            ),
            "dropout": args.dropout,
            "dropout2": args.dropout2,
            "n_params": n_params,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "protocols": details,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {metrics_path}")

    summary_path = out_dir / f"summary_{args.subject}.csv"
    write_rows(summary_path, summary_rows, SUMMARY_FIELDS)
    print(f"[Saved] {summary_path}")
    return summary_rows


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="EEGNet 时域基线 + holdout/随机 CV/LORO/LODO/run 身份负对照。"
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        default=["sub-01"],
        help="被试编号，可指定多个，如 --subject sub-01 sub-02",
    )
    parser.add_argument(
        "--task",
        default="imagine",
        choices=("imagine", "reading"),
        help="读取哪一组 CSV / run pkl",
    )
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=500.0,
        help="目标采样率；低于 500 时用 resample_poly 做抗混叠重采样",
    )
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=["cv5", "loro", "lodo"],
        choices=("holdout", "cv5", "loro", "lodo", "runid", "all"),
        help="评估协议，可给多个；默认 cv5+loro+lodo，all 表示全部五种",
    )
    parser.add_argument("--folds", type=int, default=5, help="随机 CV 折数")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（固定，不做测试集早停）")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--f1", type=int, default=8, help="EEGNet 第一层时间滤波器数")
    parser.add_argument("--d", type=int, default=2, help="EEGNet 深度乘数")
    parser.add_argument("--f2", type=int, default=16, help="EEGNet 可分离卷积输出通道数")
    parser.add_argument(
        "--kernel-seconds",
        type=float,
        default=0.5,
        help="第一层时间核长度（秒），按采样率换算成采样点",
    )
    parser.add_argument(
        "--separable-seconds",
        type=float,
        default=0.125,
        help="可分离时间核长度（秒），按采样率换算成采样点",
    )
    parser.add_argument("--dropout", type=float, default=0.25, help="第一个 dropout")
    parser.add_argument("--dropout2", type=float, default=0.5, help="第二个 dropout")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="关闭逐试次逐通道 z-score（默认开启）",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="置换次数；深度模型每个置换都要重训全部折，成本极高，默认 0",
    )
    parser.add_argument("--bootstrap", type=int, default=5000, help="run 级 bootstrap 次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="训练设备，默认自动选择 GPU",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs 至少为 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size 至少为 1")
    if args.folds < 2:
        raise SystemExit("--folds 至少为 2")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Device] {device}")

    run_name = args.run_name or f"eegnet_{datetime.now():%Y%m%d_%H%M%S}"
    subjects = list(args.subject)
    all_rows: list[dict] = []

    for subject in subjects:
        args.subject = subject
        print(f"\n===== 被试 {subject} =====")
        dataset = EEGNetDataset(
            subject=subject,
            task=args.task,
            top_k=args.top_k,
            target_sfreq=args.target_sfreq,
            normalize=not args.no_normalize,
            csv_root=args.csv_root,
            pkl_root=args.pkl_root,
        )
        out_dir = args.out_dir / run_name / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        all_rows.extend(run_subject(args, dataset, out_dir, device))
        del dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = args.out_dir / run_name / "summary_all.csv"
    write_rows(summary_path, all_rows, SUMMARY_FIELDS)
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
