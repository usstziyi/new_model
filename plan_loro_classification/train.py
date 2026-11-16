"""leave-one-run-out 句类分类：直接检验随机划分的准确率有多少来自 run 签名。

背景：``plan_baseline_lda`` 的 8:1:1 随机划分会让同一 run 的试次同时出现在
train / test；``plan_run_identity`` 已证明 run 身份可以 98% 被解码，且 run 与
句类存在明显混杂。本实验做最直接的对照：

- **leave-one-run-out（LORO）**：每次留出整个 run 作为测试集，用其余 44 个 run
  训练，句类准确率无法再借助“见过的 run 指纹”；
- **随机划分参考**：同一批特征上跑 5 折分层 CV，复现 ``plan_baseline_lda`` 的
  within-run 数字；
- **多数类基线**：每折用训练集里最多的句类去猜测试 run，给出不依赖 EEG 的下限。

如果 LORO 准确率从随机划分的 ~22.7% 掉到接近 10%，说明当前句类分类主要靠
run 特异签名，而不是跨 run 稳定的句类脑响应。

用法::

    uv run plan_loro_classification/train.py
    uv run plan_loro_classification/train.py --bands-mode abs
    uv run plan_loro_classification/train.py --model logreg
    uv run plan_loro_classification/train.py --use-64-channels
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from dataset import TOP_ROOT, TopClassEEGDataset
from model import BANDS, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "loro_classification"


def loro_evaluate(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_names: list[str],
    class_names: list[str],
    run_day: dict[int, int],
    model_name: str,
) -> dict:
    """逐 run 留出，返回每折指标和池化后的总体指标。"""
    splitter = LeaveOneGroupOut()
    fold_rows = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for train_idx, test_idx in splitter.split(feats, y, groups):
        held_out = int(groups[test_idx][0])
        model = make_model(model_name)
        model.fit(feats[train_idx], y[train_idx])
        pred = model.predict(feats[test_idx])

        present = np.unique(y[test_idx])
        counts = np.bincount(y[test_idx], minlength=len(class_names))
        train_counts = np.bincount(y[train_idx], minlength=len(class_names))
        majority = int(train_counts.argmax())

        fold_rows.append(
            {
                "run": group_names[held_out],
                "run_id": held_out,
                "day": run_day.get(held_out),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(
                    f1_score(y[test_idx], pred, labels=present, average="macro", zero_division=0)
                ),
                "majority_class": class_names[majority],
                "majority_baseline_accuracy": float(np.mean(y[test_idx] == majority)),
                "class_counts": counts.tolist(),
            }
        )
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(pred)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    accs = np.asarray([row["accuracy"] for row in fold_rows])
    f1s = np.asarray([row["macro_f1"] for row in fold_rows])
    majority_correct = sum(
        row["majority_baseline_accuracy"] * row["n_test"] for row in fold_rows
    )

    result = {
        "n_folds": len(fold_rows),
        "pooled": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "n_samples": int(len(y_true)),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=list(range(len(class_names)))
            ).tolist(),
            "per_class_recall": [
                float(np.mean(y_pred[y_true == c] == c)) if np.any(y_true == c) else None
                for c in range(len(class_names))
            ],
        },
        "fold_mean_accuracy": float(accs.mean()),
        "fold_std_accuracy": float(accs.std()),
        "fold_mean_macro_f1": float(f1s.mean()),
        "fold_std_macro_f1": float(f1s.std()),
        "majority_baseline_accuracy_pooled": float(majority_correct / len(y_true)),
        "folds": fold_rows,
    }
    return result, y_true, y_pred


def random_cv_reference(
    feats: np.ndarray,
    y: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
) -> dict:
    """同一批特征上的随机分层 CV，作为 within-run 参考。"""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for train_idx, test_idx in splitter.split(feats, y):
        model = make_model(model_name)
        model.fit(feats[train_idx], y[train_idx])
        pred = model.predict(feats[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    return {
        "folds": folds,
        "accuracy": accs,
        "macro_f1": f1s,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="leave-one-run-out 句类分类（log 频带功率 + 线性分类器）。")
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器")
    parser.add_argument("--folds", type=int, default=5, help="随机划分参考的 CV 折数")
    parser.add_argument("--seed", type=int, default=42, help="随机划分参考的种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    return parser.parse_args()


def print_fold_table(rows: list[dict]) -> None:
    header = f"{'run':<9}{'day':>4}{'n_test':>8}{'acc':>8}{'macroF1':>10}{'maj_base':>10}"
    print("\n=== 逐 run 留出结果 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['run']:<9}{row['day']:>4}{row['n_test']:>8}{row['accuracy']:>8.4f}"
            f"{row['macro_f1']:>10.4f}{row['majority_baseline_accuracy']:>10.4f}"
        )


def main():
    args = parse_args()

    dataset = TopClassEEGDataset(
        subject=args.subject,
        root=args.data_root,
        target_sfreq=args.target_sfreq,
        normalize=False,  # 相对功率尺度不变，无需逐试次 z-score
        use_64_channels=args.use_64_channels,
    )
    y = dataset.labels
    groups = dataset.groups
    class_names = list(dataset.class_names)
    num_classes = len(class_names)
    num_runs = len(dataset.group_names)
    sfreq = float(args.target_sfreq)

    run_day: dict[int, int] = {}
    for group, entry in zip(groups, dataset.meta):
        run_day.setdefault(int(group), int(entry.get("day", -1)))

    run_counts = np.array([np.sum(groups == run) for run in dataset.group_names])
    print(f"[Runs] {num_runs} 个 run | 每 run 试次 {run_counts.min()}~{run_counts.max()}")

    # feats: (N, 610)
    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    run_name = args.run_name or f"loro_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. leave-one-run-out ----------
    print(f"=== Leave-one-run-out（{num_runs} 折，{args.model}）===")
    loro, loro_y_true, loro_y_pred = loro_evaluate(
        feats, y, groups, dataset.group_names, class_names, run_day, args.model
    )
    print_fold_table(loro["folds"])

    pooled = loro["pooled"]
    print("\n=== LORO 汇总 ===")
    print(
        f"  池化 acc={pooled['accuracy']:.4f} macro_f1={pooled['macro_f1']:.4f} "
        f"weighted_f1={pooled['weighted_f1']:.4f} (n={pooled['n_samples']})"
    )
    print(
        f"  逐折 acc={loro['fold_mean_accuracy']:.4f} ± {loro['fold_std_accuracy']:.4f}   "
        f"macro_f1={loro['fold_mean_macro_f1']:.4f} ± {loro['fold_std_macro_f1']:.4f}"
    )
    print(
        f"  多数类基线 pooled acc={loro['majority_baseline_accuracy_pooled']:.4f}   "
        f"chance={1 / num_classes:.4f}"
    )
    print(f"\nLORO 池化分类报告（{args.model}）:")
    print(
        classification_report(
            loro_y_true,
            loro_y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    # ---------- 2. 随机划分参考 ----------
    reference = random_cv_reference(feats, y, args.model, args.folds, args.seed)
    print(f"=== 随机 {args.folds} 折参考（within-run，同 baseline）===")
    print(
        f"  {args.model:<7} acc={reference['accuracy_mean']:.4f} ± {reference['accuracy_std']:.4f}   "
        f"macro_f1={reference['macro_f1_mean']:.4f} ± {reference['macro_f1_std']:.4f}"
    )
    print(f"  {'chance':<7} acc={1 / num_classes:.4f}")

    # ---------- 输出 ----------
    payload = {
        "subject": args.subject,
        "task": "leave_one_run_out_classification",
        "classes": class_names,
        "n_samples": int(len(y)),
        "n_runs": num_runs,
        "n_features": int(feats.shape[1]),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": sfreq,
        "use_64_channels": args.use_64_channels,
        "loro": loro,
        "reference_random_cv": reference,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {metrics_path}")


if __name__ == "__main__":
    main()
