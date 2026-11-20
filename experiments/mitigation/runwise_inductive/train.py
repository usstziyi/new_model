"""run-wise 归一化（归纳式）下的三种评估协议：随机 5 折 CV、LORO、LODO。

与 ``experiments/diagnostics/loro_lodo`` 的唯一区别是：特征送入分类器之前先按 run 做标准化，且
μ/σ 的估计严格不使用任何留出试次（实现见 ``model.normalize_inductive``）。
特征（log 相对频带功率）与分类器（收缩 LDA）保持一致，因此结果可与
``experiments/diagnostics/baseline_lda`` 的随机划分、``experiments/diagnostics/loro_lodo`` 的留一数字直接对比。

一次运行会依次跑两种归一化口径，结果分目录存放：

- ``zscore``：减均值 + 除标准差（主结果）；
- ``centeronly``：仅减均值（σ 消融）。

输出布局为 ``<out-dir>/<run-name>/<mode>/<subject>/``（单被试时省略 ``<subject>`` 一层）。

三种协议：

1. ``cv5``：按句类分层的随机 5 折。每折内只用训练折的试次估计每个 run 的 μ/σ，
   测试试次只被变换、不参与估计；
2. ``loro``：每次留出一个 run。该 run 在训练集中不存在，只能用训练集汇总统计量
   变换，run 级偏移不会被去除；
3. ``lodo``：每次留出一个采集日，口径同 ``loro``；
4. ``runid``：run 身份负对照。把标签换成 run（45 类），只在随机 5 折下评估，
   用来确认归一化之后 run 信息是否真的被削弱（未归一化时约为 0.98）。
   ``--protocol all`` 会把它一并跑完。

每个协议都会报告池化准确率（附 run 级 cluster bootstrap 95% CI）、逐折均值与
标准差、宏平均 F1、多数类基线，以及可选的置换零分布。

用法::

    uv run experiments/mitigation/runwise_inductive/train.py --subject sub-01 --permutations 0
    uv run experiments/mitigation/runwise_inductive/train.py --protocol cv5 --subject sub-01 sub-02
    uv run experiments/mitigation/runwise_inductive/train.py --protocol loro --run-name main_8_2
    uv run experiments/mitigation/runwise_inductive/train.py --permutations 200 --run-name main_8_2
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

from dataset import CSV_ROOT, PKL_ROOT, RunwiseEEGDataset
from model import BANDS, extract_features, make_model, normalize_inductive

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "runwise_inductive"
NORMALIZATION = "runwise_inductive"
TRANSDUCTIVE = False
# (目录名, 是否只减均值)：一次运行依次跑完两种口径
MODES = (("zscore", False), ("centeronly", True))

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
]
SUMMARY_FIELDS = [
    "subject",
    "eeg_task",
    "protocol",
    "normalization",
    "transductive",
    "center_only",
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
    "permutation_n",
    "permutation_null_mean",
    "permutation_null_p95",
    "permutation_z",
    "permutation_p",
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


def combine_csvs(paths: list[Path], out_path: Path) -> None:
    """把多个同结构 CSV 拼成一个（用于跨被试汇总）。"""
    rows: list[dict] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            rows.extend(reader)
    if not rows:
        return
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_classification_csv(path: Path, subject: str, protocol: str, report: dict) -> None:
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


def evaluate_splits(
    subject: str,
    protocol: str,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    split_names: list[str],
    ci_groups: np.ndarray,
    transform,
    model_name: str,
    num_classes: int,
    seed: int,
) -> dict:
    """在给定划分上训练/预测一次，返回池化与逐折指标。"""
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    fold_rows: list[dict] = []

    for name, (train_idx, test_idx) in zip(split_names, splits):
        x_train, x_test = transform(train_idx, test_idx)
        model = make_model(model_name)
        model.fit(x_train, y[train_idx])
        pred = model.predict(x_test)

        majority = int(np.bincount(y[train_idx], minlength=num_classes).argmax())
        fold_rows.append(
            {
                "subject": subject,
                "protocol": protocol,
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
        group_parts.append(ci_groups[test_idx])

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    pooled_groups = np.concatenate(group_parts)
    ci_low, ci_high = bootstrap_accuracy_ci(y_true, y_pred, pooled_groups, seed)

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


def permutation_null(
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    transform,
    model_name: str,
    n_perm: int,
    seed: int,
    protocol: str,
) -> np.ndarray:
    """全局打乱句类标签后的池化准确率零分布。"""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    step = 10
    for i in range(n_perm):
        shuffled = rng.permutation(y)
        correct = 0
        for train_idx, test_idx in splits:
            x_train, x_test = transform(train_idx, test_idx)
            model = make_model(model_name)
            model.fit(x_train, shuffled[train_idx])
            correct += int(np.sum(model.predict(x_test) == shuffled[test_idx]))
        null[i] = correct / len(shuffled)
        if (i + 1) % step == 0 or i == n_perm - 1:
            print(
                f"[{protocol}][置换] {i + 1}/{n_perm} ({(i + 1) / n_perm:.0%}) "
                f"本次 acc={null[i]:.4f}",
                flush=True,
            )
    return null


def build_splits(
    protocol: str,
    feats: np.ndarray,
    y: np.ndarray,
    dataset: RunwiseEEGDataset,
    folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str], np.ndarray]:
    """返回 ``(splits, split_names, ci_groups)``。

    ``ci_groups`` 是 bootstrap 的重采样单元：CV5 用 run（同 run 试次相关），
    留一协议用被留出的分组本身（run 或 day）。
    """
    if protocol == "cv5":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(feats, y))
        names = [f"fold_{index}" for index in range(1, len(splits) + 1)]
        return splits, names, dataset.run_groups
    if protocol == "loro":
        splits = list(LeaveOneGroupOut().split(feats, y, dataset.run_groups))
        names = [f"run_{int(dataset.run_groups[test][0]):02d}" for _, test in splits]
        return splits, names, dataset.run_groups
    if protocol == "lodo":
        splits = list(LeaveOneGroupOut().split(feats, y, dataset.day_groups))
        names = [f"day_{int(dataset.day_groups[test][0]):02d}" for _, test in splits]
        return splits, names, dataset.day_groups
    if protocol == "runid":
        # 负对照：标签换成 run 身份，只做随机分层 CV（留一 run 对身份任务无意义）
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(feats, y))
        names = [f"fold_{index}" for index in range(1, len(splits) + 1)]
        return splits, names, dataset.run_groups
    raise ValueError(f"未知协议: {protocol}")


def print_fold_table(protocol: str, rows: list[dict]) -> None:
    header = f"{'split':<10}{'n_test':>7}{'acc':>8}{'macroF1':>9}{'maj_base':>9}"
    print(f"\n=== [{protocol}] 逐折结果 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['split']:<10}{row['n_test']:>7}{row['accuracy']:>8.4f}"
            f"{row['macro_f1']:>9.4f}{row['majority_baseline_accuracy']:>9.4f}"
        )


def run_protocol(
    protocol: str,
    feats: np.ndarray,
    y: np.ndarray,
    dataset: RunwiseEEGDataset,
    class_names: list[str],
    args,
    out_dir: Path,
    transform,
    folds: int,
    center_only: bool,
) -> tuple[dict, dict | None, dict]:
    """一个协议的完整评估：指标 + 置换 + 落盘，返回 ``(result, permutation, summary)``。"""
    splits, split_names, ci_groups = build_splits(protocol, feats, y, dataset, folds, args.seed)
    num_classes = len(class_names)
    print(f"\n----- [{protocol}] {len(splits)} 折，{args.model} -----", flush=True)

    result = evaluate_splits(
        args.subject,
        protocol,
        y,
        splits,
        split_names,
        ci_groups,
        transform,
        args.model,
        num_classes,
        args.seed,
    )
    print_fold_table(protocol, result["folds"])
    print(
        f"[{protocol}] 池化 acc={result['accuracy']:.4f} "
        f"95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
        f"macro_f1={result['macro_f1']:.4f} weighted_f1={result['weighted_f1']:.4f}"
    )
    print(
        f"[{protocol}] 逐折 acc={result['fold_mean_accuracy']:.4f} ± "
        f"{result['fold_std_accuracy']:.4f} | 多数类基线="
        f"{result['majority_baseline_accuracy']:.4f} chance={1 / num_classes:.4f}"
    )

    permutation = None
    if args.permutations > 0:
        null = permutation_null(
            y, splits, transform, args.model, args.permutations, args.seed, protocol
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
        print(
            f"[{protocol}] 置换：null={permutation['null_mean']:.4f} ± "
            f"{permutation['null_std']:.4f} p95={permutation['null_p95']:.4f} "
            f"z={permutation['z_score']:.1f} p={permutation['p_value']:.4f}"
        )
    else:
        print(f"[{protocol}] 置换已跳过（--permutations 0）")

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

    perm = permutation or {}
    summary = {
        "subject": args.subject,
        "eeg_task": args.task,
        "protocol": protocol,
        "normalization": NORMALIZATION,
        "transductive": TRANSDUCTIVE,
        "center_only": center_only,
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
        "permutation_n": perm.get("n"),
        "permutation_null_mean": perm.get("null_mean"),
        "permutation_null_p95": perm.get("null_p95"),
        "permutation_z": perm.get("z_score"),
        "permutation_p": perm.get("p_value"),
    }
    detail = {
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
    return detail, permutation, summary


def run_subject(args, out_dir: Path, center_only: bool) -> list[dict]:
    """完成一个被试在指定归一化口径下、三种协议的评估并落盘，返回 summary 行。"""
    dataset = RunwiseEEGDataset(
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

    # 归纳式归一化：每折都只用该折的训练试次估计 μ/σ
    def transform(train_idx: np.ndarray, test_idx: np.ndarray):
        return normalize_inductive(
            feats, dataset.run_groups, train_idx, test_idx, center_only=center_only
        )

    mode_text = "仅减均值" if center_only else "减均值 + 除标准差"
    print(f"[Normalize] 归纳式 run-wise（{mode_text}），μ/σ 只来自训练试次", flush=True)

    protocols = ["cv5", "loro", "lodo", "runid"] if args.protocol == "all" else [args.protocol]
    details: dict[str, dict] = {}
    summary_rows: list[dict] = []

    # run 身份负对照：标签换成 run，仅用于随机 CV
    run_ids = sorted({int(run) for run in dataset.run_groups})
    run_to_label = {run: index for index, run in enumerate(run_ids)}
    y_run = np.asarray([run_to_label[int(run)] for run in dataset.run_groups], dtype=np.int64)
    run_names = [f"run_{run:02d}" for run in run_ids]
    run_folds = min(args.folds, int(np.bincount(y_run).min()))
    print(
        f"[RunIdentity] {len(run_ids)} 类（最小 run {int(np.bincount(y_run).min())} 条，"
        f"{run_folds} 折），chance={1 / len(run_ids):.4f}",
        flush=True,
    )

    for protocol in protocols:
        if protocol == "runid":
            task_y, task_names, task_folds = y_run, run_names, run_folds
        else:
            task_y, task_names, task_folds = y, class_names, folds
        detail, _, summary = run_protocol(
            protocol,
            feats,
            task_y,
            dataset,
            task_names,
            args,
            out_dir,
            transform,
            task_folds,
            center_only,
        )
        details[protocol] = detail
        summary_rows.append(summary)

    print("\n=== 协议对比（池化 acc）===")
    for name, detail in details.items():
        print(
            f"  {name:<5} acc={detail['accuracy']:.4f} "
            f"95% CI [{detail['ci'][0]:.3f}, {detail['ci'][1]:.3f}] "
            f"macro_f1={detail['macro_f1']:.4f} | 逐折 "
            f"{detail['fold_mean_accuracy']:.4f} ± {detail['fold_std_accuracy']:.4f}"
        )

    payload = {
        "subject": args.subject,
        "task": "runwise_inductive_classification",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "normalization": NORMALIZATION,
        "transductive": TRANSDUCTIVE,
        "center_only": center_only,
        "classes": class_names,
        "n_samples": int(len(y)),
        "n_features": int(feats.shape[1]),
        "bands": {key: list(value) for key, value in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": sfreq,
        "use_64_channels": args.use_64_channels,
        "cv_folds": folds,
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
        description="run-wise 归一化（归纳式）+ 随机 CV / LORO / LODO 句类分类。"
    )
    parser.add_argument("--subject", nargs="+", default=["sub-01"], help="被试编号，可指定多个")
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"), help="读取哪一组 CSV / run pkl")
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器")
    parser.add_argument(
        "--protocol",
        default="all",
        choices=("cv5", "loro", "lodo", "runid", "all"),
        help="跑哪个协议：随机 5 折 / 留一 run / 留一天 / run 身份负对照 / 全部",
    )
    parser.add_argument("--folds", type=int, default=5, help="随机 CV 折数")
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="置换次数，0 表示跳过。LORO 下每轮需要 45 次拟合，成本较高",
    )
    parser.add_argument("--seed", type=int, default=42, help="CV、bootstrap 与置换的随机种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = list(args.subject)
    base_folds = args.folds
    run_name = args.run_name or f"runwise_inductive_{datetime.now():%Y%m%d_%H%M%S}"

    for mode_name, center_only in MODES:
        print(f"\n##### 归一化口径：{mode_name} #####")
        all_rows: list[dict] = []
        for subject in subjects:
            args.subject = subject
            args.folds = base_folds  # 上一个被试可能因样本太少降过折数，重置
            out_dir = args.out_dir / run_name / mode_name
            if len(subjects) > 1:
                out_dir = out_dir / subject
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n===== 被试 {subject} =====")
            all_rows.extend(run_subject(args, out_dir, center_only))

        if len(subjects) > 1:
            summary_path = args.out_dir / run_name / mode_name / "summary_all.csv"
            write_rows(summary_path, all_rows, SUMMARY_FIELDS)
            print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
