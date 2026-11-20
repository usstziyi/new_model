"""正交化（归纳式）：cv5 / LORO 下逐 k 评估 run 方向投影对句类解码的影响。

方法：在每个训练折上拟合 run 身份 LDA，取判别力最强的 ``k`` 个方向，正交化后
把特征投影到其正交补空间，再训练句类分类器。归纳式指 run 分类器只用训练
试次拟合，留出 run 不参与方向估计。

评估矩阵：协议 ∈ {cv5, loro}，k ∈ {1, 5, 10, 44}（实际可用方向数不足时自动
截断，并在输出中记录 ``k_effective`` 与 ``n_run_directions``）。不跑 LODO，
也不跑 run 身份负对照。

输出布局::

    <out-dir>/<run-name>/<subject>/
        summary_<subject>.csv
        folds_<protocol>_k<k>_<subject>.csv
        classification_report_<protocol>_k<k>_<subject>.csv
        metrics.json

多被试时在 ``<run-name>/`` 下额外汇总 ``summary_all.csv``。

用法::

    uv run experiments/mitigation/orthogonalization_inductive/train.py
    uv run experiments/mitigation/orthogonalization_inductive/train.py --subject sub-01
    uv run experiments/mitigation/orthogonalization_inductive/train.py --protocol cv5 --ks 1 10
    uv run experiments/mitigation/orthogonalization_inductive/train.py --protocol loro --ks 10 44
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from dataset import CSV_ROOT, PKL_ROOT, OrthogonalizationEEGDataset
from model import (
    BANDS,
    directions_from_lda,
    extract_features,
    fit_run_lda,
    make_model,
    project,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "orthogonalization_inductive"
NORMALIZATION = "orthogonalization"
TRANSDUCTIVE = False
DEFAULT_KS = (1, 5, 10, 44)
DEFAULT_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]

REPORT_FIELDS = ["subject", "protocol", "k", "label", "precision", "recall", "f1", "support"]
FOLD_FIELDS = [
    "subject",
    "protocol",
    "k",
    "k_effective",
    "n_run_directions",
    "n_train_runs",
    "split",
    "n_train",
    "n_test",
    "accuracy",
    "macro_f1",
    "majority_baseline_accuracy",
]
SUMMARY_FIELDS = [
    "subject",
    "eeg_task",
    "protocol",
    "normalization",
    "transductive",
    "k",
    "k_effective",
    "n_run_directions",
    "n_train_runs",
    "top_k",
    "n_samples",
    "n_splits",
    "n_features",
    "model",
    "band_mode",
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
]


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
    path: Path,
    subject: str,
    protocol: str,
    k: int,
    report: dict,
) -> None:
    """把池化 ``classification_report(output_dict=True)`` 写成逐类 CSV。"""
    total_support = sum(
        metrics["support"]
        for label, metrics in report.items()
        if isinstance(metrics, dict) and label not in ("macro avg", "weighted avg", "samples avg")
    )
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {
                "subject": subject,
                "protocol": protocol,
                "k": k,
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
                "k": k,
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
    y_true,
    y_pred,
    groups,
    seed: int,
    n_boot: int = 5000,
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


def build_splits(protocol: str, y: np.ndarray, dataset, folds: int, seed: int):
    """返回 ``(splits, split_names)``：cv5 分层随机划分，loro 留一 run。"""
    if protocol == "cv5":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
        names = [f"fold_{index}" for index in range(1, len(splits) + 1)]
        return splits, names
    if protocol == "loro":
        splits = list(LeaveOneGroupOut().split(np.zeros(len(y)), y, dataset.run_groups))
        names = [f"run_{int(dataset.run_groups[test][0]):02d}" for _, test in splits]
        return splits, names
    raise ValueError(f"未知协议: {protocol}")


def evaluate_inductive(
    protocol: str,
    k: int,
    feats: np.ndarray,
    y: np.ndarray,
    dataset,
    fold_models,
    args,
    num_classes: int,
) -> dict:
    """归纳式：每折只用训练试次拟合 run 方向，再投影并训练句类分类器。

    ``fold_models`` 由 ``run_subject`` 预先拟合好，元素为
    ``(split_name, train_idx, test_idx, scaler, lda)``；同一个 run 分类器
    在多个 ``k`` 之间复用，保证 k 扫描使用的是同一组判别方向排序。
    """
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    fold_rows: list[dict] = []
    k_effective_values: list[int] = []
    n_direction_values: list[int] = []
    n_train_runs_values: list[int] = []

    for name, train_idx, test_idx, scaler, lda in fold_models:
        train_groups = dataset.run_groups[train_idx]
        directions, k_effective, n_available = directions_from_lda(lda, k)

        x_train = project(feats[train_idx], scaler, directions)
        x_test = project(feats[test_idx], scaler, directions)

        model = make_model(args.model)
        model.fit(x_train, y[train_idx])
        pred = model.predict(x_test)

        majority = int(np.bincount(y[train_idx], minlength=num_classes).argmax())
        fold_rows.append(
            {
                "subject": args.subject,
                "protocol": protocol,
                "k": k,
                "k_effective": k_effective,
                "n_run_directions": n_available,
                "n_train_runs": int(len(np.unique(train_groups))),
                "split": name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                "majority_baseline_accuracy": float(np.mean(y[test_idx] == majority)),
            }
        )
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(pred)
        group_parts.append(dataset.run_groups[test_idx])
        k_effective_values.append(k_effective)
        n_direction_values.append(n_available)
        n_train_runs_values.append(int(len(np.unique(train_groups))))

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
        "k": k,
        "k_effective": int(min(k_effective_values)),
        "n_run_directions": int(min(n_direction_values)),
        "n_train_runs": int(min(n_train_runs_values)),
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
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


def run_subject(args, out_dir: Path) -> list[dict]:
    """完成一个被试在 cv5 / LORO 下所有 k 的评估并落盘，返回 summary 行。"""
    dataset = OrthogonalizationEEGDataset(
        subject=args.subject,
        task=args.task,
        top_k=args.top_k,
        csv_root=args.csv_root,
        pkl_root=args.pkl_root,
        target_sfreq=args.target_sfreq,
        normalize=False,  # 相对功率尺度不变，无需逐试次 z-score
        use_64_channels=args.use_64_channels,
    )
    y = dataset.labels
    class_names = list(dataset.class_names)
    num_classes = len(class_names)
    sfreq = float(args.target_sfreq)

    folds = args.folds
    min_class = int(np.bincount(y).min())
    if folds > min_class:
        print(f"[Warn] 最小的句类只有 {min_class} 条，CV 折数从 {folds} 降到 {min_class}")
        folds = min_class

    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    protocols = ["cv5", "loro"] if args.protocol == "all" else [args.protocol]
    details: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for protocol in protocols:
        splits, split_names = build_splits(protocol, y, dataset, folds, args.seed)
        details[protocol] = {}
        print(f"\n===== [{protocol}] {len(splits)} 折 =====", flush=True)

        # 每折只拟合一次 run LDA；不同 k 复用同一组判别方向
        fold_models = []
        for name, (train_idx, test_idx) in zip(split_names, splits):
            scaler, lda = fit_run_lda(feats[train_idx], dataset.run_groups[train_idx])
            fold_models.append((name, train_idx, test_idx, scaler, lda))
        print(
            f"[Directions] {protocol}: {len(fold_models)} 折的 run LDA 已拟合，"
            f"训练 run 数范围 "
            f"{min(len(np.unique(dataset.run_groups[train_idx])) for _, train_idx, _, _, _ in fold_models)}"
            f"–{max(len(np.unique(dataset.run_groups[train_idx])) for _, train_idx, _, _, _ in fold_models)}",
            flush=True,
        )

        for k in args.ks:
            result = evaluate_inductive(
                protocol,
                k,
                feats,
                y,
                dataset,
                fold_models,
                args,
                num_classes,
            )
            print(
                f"[{protocol}][k={k}] k_effective={result['k_effective']} "
                f"n_dirs={result['n_run_directions']} | 池化 acc={result['accuracy']:.4f} "
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
                target_names=class_names,
                digits=4,
                zero_division=0,
                output_dict=True,
            )
            save_classification_csv(
                out_dir / f"classification_report_{protocol}_k{k}_{args.subject}.csv",
                args.subject,
                protocol,
                k,
                report,
            )
            write_rows(
                out_dir / f"folds_{protocol}_k{k}_{args.subject}.csv",
                result["folds"],
                FOLD_FIELDS,
            )

            summary_rows.append(
                {
                    "subject": args.subject,
                    "eeg_task": args.task,
                    "protocol": protocol,
                    "normalization": NORMALIZATION,
                    "transductive": TRANSDUCTIVE,
                    "k": k,
                    "k_effective": result["k_effective"],
                    "n_run_directions": result["n_run_directions"],
                    "n_train_runs": result["n_train_runs"],
                    "top_k": args.top_k,
                    "n_samples": result["n_samples"],
                    "n_splits": result["n_splits"],
                    "n_features": int(feats.shape[1]),
                    "model": args.model,
                    "band_mode": args.bands_mode,
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
                }
            )
            details[protocol][str(k)] = {
                "k_effective": result["k_effective"],
                "n_run_directions": result["n_run_directions"],
                "n_train_runs": result["n_train_runs"],
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
            }

    payload = {
        "subject": args.subject,
        "task": "orthogonalization_inductive_classification",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "normalization": NORMALIZATION,
        "transductive": TRANSDUCTIVE,
        "classes": class_names,
        "n_samples": int(len(y)),
        "n_features": int(feats.shape[1]),
        "bands": {key: list(value) for key, value in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": sfreq,
        "use_64_channels": args.use_64_channels,
        "cv_folds": folds,
        "ks": list(args.ks),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="run 判别方向正交化（归纳式）+ 随机 CV / LORO 句类分类。"
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="被试编号，可指定多个",
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
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument(
        "--bands-mode",
        default="rel",
        choices=("rel", "abs"),
        help="相对/绝对频带功率",
    )
    parser.add_argument(
        "--model",
        default="lda",
        choices=("lda", "logreg", "svm"),
        help="句类分类器",
    )
    parser.add_argument(
        "--protocol",
        default="all",
        choices=("cv5", "loro", "all"),
        help="跑哪个协议：随机 5 折 / 留一 run / 全部",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_KS),
        help="要去掉的 run 判别方向数，可给多个；超过可用方向数时自动截断",
    )
    parser.add_argument("--folds", type=int, default=5, help="随机 CV 折数")
    parser.add_argument("--bootstrap", type=int, default=5000, help="run 级 bootstrap 次数")
    parser.add_argument("--seed", type=int, default=42, help="CV、bootstrap 的随机种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    return parser.parse_args()


def main():
    args = parse_args()
    ks = sorted({int(k) for k in args.ks if int(k) > 0})
    if not ks:
        raise SystemExit("--ks 至少需要一个正整数")
    args.ks = ks

    subjects = list(args.subject)
    run_name = args.run_name or f"orthogonalization_inductive_{datetime.now():%Y%m%d_%H%M%S}"

    all_rows: list[dict] = []
    for subject in subjects:
        args.subject = subject
        out_dir = args.out_dir / run_name
        if len(subjects) > 1:
            out_dir = out_dir / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== 被试 {subject} =====", flush=True)
        all_rows.extend(run_subject(args, out_dir))

    if len(subjects) > 1:
        summary_path = args.out_dir / run_name / "summary_all.csv"
        write_rows(summary_path, all_rows, SUMMARY_FIELDS)
        print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
