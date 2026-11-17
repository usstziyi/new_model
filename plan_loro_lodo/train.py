"""leave-one-run-out / leave-one-day-out 句类分类：量化随机划分的准确率有多少来自 run / day 签名。

背景：``plan_baseline_lda`` 的随机划分会让同一 run 的试次同时出现在
train / test；``plan_run_identity`` 已证明 run 身份可以接近 100% 被解码，
且 run 与句类明显混杂。本实验在**同一份数据、同一套特征与分类器**上做两组
留一对照：

1. **leave-one-run-out（LORO）**：每次留出整个 run 作为测试集，用其余 run 训练，
   句类准确率无法再借助“见过的 run 指纹”；
2. **leave-one-day-out（LODO）**：每次留出整整一个采集日，进一步排除日间漂移；
3. **随机 5 折参考**：与 ``plan_baseline_lda`` 的 CV 口径一致，并保留每折 OOF
   预测，用于**逐组配对比较**（参考准确率 − 留一准确率）；
4. **多数类基线 / chance**：不依赖 EEG 的下限；留一池化准确率附组级 bootstrap
   95% CI；
5. **置换零分布**（可选，默认 200 次）：全局打乱句类标签后重跑留一，给出 p 值；
6. **频段扫描**（可选）：``all`` / 单频段 / 留一频段共 11 种特征子集；扫描模式
   不做置换，避免 11 × 45 × B 次拟合。

注意：若同时报告多个被试或多个频段配置的 p 值，需要在论文中说明多重比较校正方法
（如 Bonferroni 或 FDR）。

输出：每个被试一份 ``metrics.json``，以及
``classification_report_{loro,lodo}_<subject>.csv``、``summary_<subject>.csv``、
``paired_{loro,lodo}_<subject>.csv``；``--leave-out both`` 时额外输出
``run_vs_day_<subject>.csv``（逐日配对 LORO − LODO）。多被试时每个被试一个子目录，
并在运行根目录汇总 ``summary_all.csv``。

注意两种“差值”口径不同：``pooled_diff_reference_minus_loo`` 按试次数加权，
``paired_mean_diff`` 按组等权（每个 run/day 一个数，再取平均）；两者不相等是正常的，
审稿时按问题选择，不要混用。

用法::

    uv run plan_loro_lodo/train.py
    uv run plan_loro_lodo/train.py --subject sub-01 sub-02
    uv run plan_loro_lodo/train.py --leave-out day
    uv run plan_loro_lodo/train.py --permutations 0
    uv run plan_loro_lodo/train.py --sweep-bands --permutations 0
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from dataset import CSV_ROOT, PKL_ROOT, LoroLodoEEGDataset
from model import BANDS, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "loro_lodo"

MODE_LABELS = {"run": "loro", "day": "lodo"}

REPORT_FIELDS = ["subject", "mode", "label", "precision", "recall", "f1", "support"]
PAIRED_FIELDS = [
    "subject",
    "mode",
    "group",
    "day",
    "n_test",
    "loo_accuracy",
    "reference_accuracy",
    "diff_reference_minus_loo",
    "majority_baseline_accuracy",
]
SUMMARY_FIELDS = [
    "subject",
    "eeg_task",
    "top_k",
    "mode",
    "n_groups",
    "reference_cv_folds",
    "n_samples",
    "n_features",
    "model",
    "band_mode",
    "chance",
    "pooled_accuracy",
    "pooled_ci_low",
    "pooled_ci_high",
    "pooled_macro_f1",
    "pooled_weighted_f1",
    "fold_mean_accuracy",
    "fold_std_accuracy",
    "fold_mean_macro_f1",
    "fold_std_macro_f1",
    "majority_baseline_accuracy",
    "reference_cv_accuracy_mean",
    "reference_cv_accuracy_std",
    "reference_cv_macro_f1_mean",
    "reference_cv_macro_f1_std",
    "reference_cv_pooled_accuracy",
    "reference_cv_pooled_macro_f1",
    "pooled_diff_reference_minus_loo",
    "paired_mean_diff",
    "paired_median_diff",
    "paired_ci_low",
    "paired_ci_high",
    "paired_wilcoxon_p",
    "paired_n",
    "permutation_n",
    "permutation_null_mean",
    "permutation_null_p95",
    "permutation_z",
    "permutation_p",
]
SWEEP_FIELDS = [
    "subject",
    "config",
    "mode",
    "n_features",
    "loo_accuracy",
    "loo_ci_low",
    "loo_ci_high",
    "loo_macro_f1",
    "fold_mean_accuracy",
    "fold_std_accuracy",
    "reference_cv_accuracy",
    "paired_mean_diff",
    "majority_baseline_accuracy",
    "chance",
]


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


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """把字典列表写成 UTF-8-SIG CSV（Excel 可直接打开）。"""
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_classification_csv(path: Path, subject: str, mode_label: str, report: dict) -> None:
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
                "mode": mode_label,
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
        elif label == "accuracy":  # output_dict 模式下 accuracy 是标量
            row = {
                "subject": subject,
                "mode": mode_label,
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
    """按组（run / day）重采样的准确率 95% CI（cluster bootstrap）。"""
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


def leave_one_group_out(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_names: list[str],
    class_names: list[str],
    model_name: str,
    seed: int,
):
    """逐组留出，返回指标、池化预测与逐组准确率。"""
    splitter = LeaveOneGroupOut()
    name_by_id = dict(zip(np.unique(groups).tolist(), group_names))
    fold_rows = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []

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
                "group": name_by_id[held_out],
                "group_id": held_out,
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
        group_parts.append(groups[test_idx])

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    pooled_groups = np.concatenate(group_parts)
    ci_low, ci_high = bootstrap_accuracy_ci(y_true, y_pred, pooled_groups, seed)
    accs = np.asarray([row["accuracy"] for row in fold_rows])
    f1s = np.asarray([row["macro_f1"] for row in fold_rows])
    majority_correct = sum(
        row["majority_baseline_accuracy"] * row["n_test"] for row in fold_rows
    )

    result = {
        "n_groups": len(fold_rows),
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
        "pooled_accuracy_ci": [ci_low, ci_high],
        "per_group_accuracy": {row["group_id"]: row["accuracy"] for row in fold_rows},
        "folds": fold_rows,
    }
    return result, y_true, y_pred


def random_cv_reference(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
) -> dict:
    """与 baseline 口径一致的随机分层 CV，并保留 OOF 预测用于逐组配对。"""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=y.dtype)
    accs, f1s = [], []
    for train_idx, test_idx in splitter.split(feats, y):
        model = make_model(model_name)
        model.fit(feats[train_idx], y[train_idx])
        pred = model.predict(feats[test_idx])
        oof[test_idx] = pred
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))

    per_group = {
        int(group): float(accuracy_score(y[groups == group], oof[groups == group]))
        for group in np.unique(groups)
    }
    return {
        "folds": folds,
        "accuracy": [float(value) for value in accs],
        "macro_f1": [float(value) for value in f1s],
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "pooled_accuracy": float(accuracy_score(y, oof)),
        "pooled_macro_f1": float(f1_score(y, oof, average="macro", zero_division=0)),
        "per_group_accuracy": per_group,
    }


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    """对逐组差值做 bootstrap，返回均值的 95% 百分位区间。"""
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(
    group_ids: list[int],
    reference: dict[int, float],
    leave_one_out: dict[int, float],
    seed: int,
) -> dict:
    """逐组配对：随机参考准确率 − 留一准确率。"""
    diffs = np.asarray(
        [reference[int(group)] - leave_one_out[int(group)] for group in group_ids],
        dtype=float,
    )
    ci_low, ci_high = _bootstrap_ci(diffs, seed)
    try:
        p_value = float(wilcoxon(diffs).pvalue)
    except ValueError:  # 全部差值为 0 时无法检验
        p_value = None
    return {
        "n_pairs": int(len(diffs)),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": p_value,
        "per_group": {
            int(group): float(diff) for group, diff in zip(group_ids, diffs)
        },
    }


def permutation_null(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_perm: int,
    seed: int,
    mode_label: str = "",
) -> np.ndarray:
    """全局打乱句类标签后的留一池化准确率零分布。"""
    rng = np.random.default_rng(seed)
    splitter = LeaveOneGroupOut()
    splits = list(splitter.split(feats, y, groups))  # 只依赖 groups，可复用
    null = np.empty(n_perm)
    step = 10  # 每 10 次置换打印一行进度
    tag = f"[{mode_label}] " if mode_label else ""
    for i in range(n_perm):
        shuffled = rng.permutation(y)
        correct = 0
        for train_idx, test_idx in splits:
            model = make_model(model_name)
            model.fit(feats[train_idx], shuffled[train_idx])
            correct += int(np.sum(model.predict(feats[test_idx]) == shuffled[test_idx]))
        null[i] = correct / len(shuffled)
        if (i + 1) % step == 0 or i == n_perm - 1:
            print(
                f"{tag}[置换] {i + 1}/{n_perm} ({(i + 1) / n_perm:.0%}) "
                f"本次 acc={null[i]:.4f}",
                flush=True,
            )
    return null


def evaluate_mode(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_ids: list[int],
    group_names: list[str],
    class_names: list[str],
    model_name: str,
    folds: int,
    seed: int,
    n_perm: int,
    mode_label: str = "",
):
    """一个分组模式（run / day）的完整评估。"""
    loo, y_true, y_pred = leave_one_group_out(
        feats, y, groups, group_names, class_names, model_name, seed
    )
    reference = random_cv_reference(feats, y, groups, model_name, folds, seed)
    paired = paired_comparison(
        group_ids, reference["per_group_accuracy"], loo["per_group_accuracy"], seed
    )

    permutation = None
    if n_perm > 0:
        null = permutation_null(feats, y, groups, model_name, n_perm, seed, mode_label)
        observed = float(loo["pooled"]["accuracy"])
        permutation = {
            "n": n_perm,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "observed": observed,
            "z_score": float((observed - null.mean()) / (null.std() + 1e-12)),
            "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "null_samples": null.tolist(),
        }
    return loo, reference, paired, permutation, y_true, y_pred


def compare_run_vs_day(
    y: np.ndarray,
    day_groups: np.ndarray,
    day_ids: list[int],
    loro_pred: np.ndarray,
    lodo_pred: np.ndarray,
    seed: int,
) -> dict:
    """逐日配对：LORO 准确率 − LODO 准确率（两种预测都覆盖全部试次一次）。"""
    rows = []
    for day in day_ids:
        mask = day_groups == day
        rows.append(
            {
                "day": day,
                "n_test": int(mask.sum()),
                "loro_accuracy": float(accuracy_score(y[mask], loro_pred[mask])),
                "lodo_accuracy": float(accuracy_score(y[mask], lodo_pred[mask])),
            }
        )
        rows[-1]["diff_loro_minus_lodo"] = (
            rows[-1]["loro_accuracy"] - rows[-1]["lodo_accuracy"]
        )

    diffs = np.asarray([row["diff_loro_minus_lodo"] for row in rows], dtype=float)
    ci_low, ci_high = _bootstrap_ci(diffs, seed)
    try:
        p_value = float(wilcoxon(diffs).pvalue)
    except ValueError:
        p_value = None
    return {
        "rows": rows,
        "mean_diff": float(diffs.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": p_value,
    }


def print_fold_table(mode_label: str, rows: list[dict]) -> None:
    header = f"{'group':<10}{'n_test':>7}{'acc':>8}{'macroF1':>9}{'maj_base':>9}"
    print(f"\n=== [{mode_label}] 逐组留出结果 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['group']:<10}{row['n_test']:>7}{row['accuracy']:>8.4f}"
            f"{row['macro_f1']:>9.4f}{row['majority_baseline_accuracy']:>9.4f}"
        )


def summarize_mode(
    subject: str,
    eeg_task: str,
    top_k: int,
    mode_label: str,
    loo: dict,
    reference: dict,
    paired: dict,
    permutation: dict | None,
    n_features: int,
    n_samples: int,
    model_name: str,
    band_mode: str,
    chance: float,
    reference_folds: int,
) -> dict:
    """写 summary CSV 的一行（所有数值都在这里落盘，便于跨被试汇总）。"""
    perm = permutation or {}
    return {
        "subject": subject,
        "eeg_task": eeg_task,
        "top_k": top_k,
        "mode": mode_label,
        "n_groups": loo["n_groups"],
        "reference_cv_folds": reference_folds,
        "n_samples": n_samples,
        "n_features": n_features,
        "model": model_name,
        "band_mode": band_mode,
        "chance": chance,
        "pooled_accuracy": loo["pooled"]["accuracy"],
        "pooled_ci_low": loo["pooled_accuracy_ci"][0],
        "pooled_ci_high": loo["pooled_accuracy_ci"][1],
        "pooled_macro_f1": loo["pooled"]["macro_f1"],
        "pooled_weighted_f1": loo["pooled"]["weighted_f1"],
        "fold_mean_accuracy": loo["fold_mean_accuracy"],
        "fold_std_accuracy": loo["fold_std_accuracy"],
        "fold_mean_macro_f1": loo["fold_mean_macro_f1"],
        "fold_std_macro_f1": loo["fold_std_macro_f1"],
        "majority_baseline_accuracy": loo["majority_baseline_accuracy_pooled"],
        "reference_cv_accuracy_mean": reference["accuracy_mean"],
        "reference_cv_accuracy_std": reference["accuracy_std"],
        "reference_cv_macro_f1_mean": reference["macro_f1_mean"],
        "reference_cv_macro_f1_std": reference["macro_f1_std"],
        "reference_cv_pooled_accuracy": reference["pooled_accuracy"],
        "reference_cv_pooled_macro_f1": reference["pooled_macro_f1"],
        "pooled_diff_reference_minus_loo": (
            reference["pooled_accuracy"] - loo["pooled"]["accuracy"]
        ),
        "paired_mean_diff": paired["mean_diff"],
        "paired_median_diff": paired["median_diff"],
        "paired_ci_low": paired["ci_low"],
        "paired_ci_high": paired["ci_high"],
        "paired_wilcoxon_p": paired["wilcoxon_p"],
        "paired_n": paired["n_pairs"],
        "permutation_n": perm.get("n"),
        "permutation_null_mean": perm.get("null_mean"),
        "permutation_null_p95": perm.get("null_p95"),
        "permutation_z": perm.get("z_score"),
        "permutation_p": perm.get("p_value"),
    }


def save_paired_csv(
    path: Path,
    subject: str,
    mode_label: str,
    loo: dict,
    reference: dict,
    run_day: dict[int, int],
) -> None:
    rows = []
    for fold in loo["folds"]:
        group = int(fold["group_id"])
        ref_acc = reference["per_group_accuracy"][group]
        rows.append(
            {
                "subject": subject,
                "mode": mode_label,
                "group": fold["group"],
                "day": run_day[group] if mode_label == "loro" else group,
                "n_test": fold["n_test"],
                "loo_accuracy": fold["accuracy"],
                "reference_accuracy": ref_acc,
                "diff_reference_minus_loo": ref_acc - fold["accuracy"],
                "majority_baseline_accuracy": fold["majority_baseline_accuracy"],
            }
        )
    write_rows(path, rows, PAIRED_FIELDS)


def band_columns(only, exclude, channels_per_band: int) -> tuple[np.ndarray, list[str]]:
    """返回按 ``BANDS`` 顺序排列的列索引，以及实际保留的频段名。"""
    bands = list(BANDS)
    kept = [
        name
        for name in bands
        if (not only or name in only) and name not in (exclude or [])
    ]
    if not kept:
        raise SystemExit("频段选择结果为空")

    index = []
    for name in kept:
        start = bands.index(name) * channels_per_band
        index.append(np.arange(start, start + channels_per_band))
    return np.concatenate(index), kept


def sweep_configs(channels_per_band: int) -> list[tuple[str, np.ndarray]]:
    """all + 5 个单频段 + 5 个留一频段，共 11 种列选择。"""
    bands = list(BANDS)
    configs = [("all", np.arange(len(bands) * channels_per_band))]
    configs += [(f"only:{name}", band_columns([name], None, channels_per_band)[0]) for name in bands]
    configs += [(f"drop:{name}", band_columns(None, [name], channels_per_band)[0]) for name in bands]
    return configs


def run_sweep(
    feats: np.ndarray,
    y: np.ndarray,
    modes: dict,
    selected_modes: list[str],
    class_names: list[str],
    args,
    folds: int,
    out_dir: Path,
) -> None:
    """逐频段评估 LORO / LODO；扫描模式不做置换以控制成本。"""
    rows = []
    for label, columns in sweep_configs(feats.shape[1] // len(BANDS)):
        print(f"\n=== [{label}] {len(columns)} 维 ===", flush=True)
        for mode in selected_modes:
            spec = modes[mode]
            loo, _, _ = leave_one_group_out(
                feats[:, columns], y, spec["groups"], spec["names"], class_names, args.model, args.seed
            )
            reference = random_cv_reference(
                feats[:, columns], y, spec["groups"], args.model, folds, args.seed
            )
            paired = paired_comparison(
                spec["ids"], reference["per_group_accuracy"], loo["per_group_accuracy"], args.seed
            )
            row = {
                "subject": args.subject,
                "config": label,
                "mode": MODE_LABELS[mode],
                "n_features": int(columns.size),
                "loo_accuracy": loo["pooled"]["accuracy"],
                "loo_ci_low": loo["pooled_accuracy_ci"][0],
                "loo_ci_high": loo["pooled_accuracy_ci"][1],
                "loo_macro_f1": loo["pooled"]["macro_f1"],
                "fold_mean_accuracy": loo["fold_mean_accuracy"],
                "fold_std_accuracy": loo["fold_std_accuracy"],
                "reference_cv_accuracy": reference["accuracy_mean"],
                "paired_mean_diff": paired["mean_diff"],
                "majority_baseline_accuracy": loo["majority_baseline_accuracy_pooled"],
                "chance": float(1 / len(class_names)),
            }
            rows.append(row)
            print(
                f"  {row['mode']}: loo={row['loo_accuracy']:.4f} "
                f"95% CI [{row['loo_ci_low']:.3f},{row['loo_ci_high']:.3f}] "
                f"macro_f1={row['loo_macro_f1']:.4f} ref_cv={row['reference_cv_accuracy']:.4f} "
                f"paired_diff={row['paired_mean_diff']:+.4f}",
                flush=True,
            )

    write_rows(out_dir / f"sweep_{args.subject}.csv", rows, SWEEP_FIELDS)
    payload = {
        "subject": args.subject,
        "task": "loro_lodo_classification_sweep",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "band_mode": args.bands_mode,
        "model": args.model,
        "chance": float(1 / len(class_names)),
        "sweep": rows,
    }
    path = out_dir / f"sweep_{args.subject}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {path}")


def build_modes(dataset: LoroLodoEEGDataset) -> dict:
    return {
        "run": {
            "groups": dataset.run_groups,
            "ids": list(dataset.run_ids),
            "names": list(dataset.run_names),
        },
        "day": {
            "groups": dataset.day_groups,
            "ids": list(dataset.day_ids),
            "names": list(dataset.day_names),
        },
    }


def run_subject(args, out_dir: Path) -> list[dict]:
    """完成一个被试的 LORO / LODO 评估并落盘，返回 summary 行。"""
    dataset = LoroLodoEEGDataset(
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
    modes = build_modes(dataset)
    num_classes = len(class_names)
    sfreq = float(args.target_sfreq)

    run_day: dict[int, int] = {}
    for run, day in zip(dataset.run_groups, dataset.day_groups):
        run_day.setdefault(int(run), int(day))

    folds = args.folds
    min_class = int(np.bincount(y).min())
    if folds > min_class:
        print(f"[Warn] 最小的句类只有 {min_class} 条，参考 CV 折数从 {folds} 降到 {min_class}")
        folds = min_class

    run_counts = np.array([np.sum(dataset.run_groups == run) for run in dataset.run_ids])
    day_counts = np.array([np.sum(dataset.day_groups == day) for day in dataset.day_ids])
    print(
        f"[Groups] {len(dataset.run_ids)} 个 run（每 run {run_counts.min()}~{run_counts.max()} 条）| "
        f"{len(dataset.day_ids)} 个 day（每 day {day_counts.min()}~{day_counts.max()} 条）| "
        f"最小句类 {min_class} 条"
    )

    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    selected_modes = ["run", "day"] if args.leave_out == "both" else [args.leave_out]

    if args.sweep_bands:
        run_sweep(feats, y, modes, selected_modes, class_names, args, folds, out_dir)
        return []

    results = {}
    predictions = {}
    summary_rows = []
    for mode in selected_modes:
        spec = modes[mode]
        label = MODE_LABELS[mode]
        print(f"\n----- {label}（{len(spec['ids'])} 折，{args.model}）-----", flush=True)
        loo, reference, paired, permutation, y_true, y_pred = evaluate_mode(
            feats,
            y,
            spec["groups"],
            spec["ids"],
            spec["names"],
            class_names,
            args.model,
            folds,
            args.seed,
            args.permutations,
            label,
        )
        print_fold_table(label, loo["folds"])
        pooled = loo["pooled"]
        print(
            f"\n[{label}] 池化 acc={pooled['accuracy']:.4f} "
            f"95% CI [{loo['pooled_accuracy_ci'][0]:.4f}, {loo['pooled_accuracy_ci'][1]:.4f}] "
            f"macro_f1={pooled['macro_f1']:.4f} "
            f"weighted_f1={pooled['weighted_f1']:.4f} (n={pooled['n_samples']})"
        )
        print(
            f"[{label}] 逐折 acc={loo['fold_mean_accuracy']:.4f} ± {loo['fold_std_accuracy']:.4f} "
            f"macro_f1={loo['fold_mean_macro_f1']:.4f} ± {loo['fold_std_macro_f1']:.4f}"
        )
        print(
            f"[{label}] 多数类基线={loo['majority_baseline_accuracy_pooled']:.4f} "
            f"chance={1 / num_classes:.4f}"
        )
        p_text = "-" if paired["wilcoxon_p"] is None else f"{paired['wilcoxon_p']:.4g}"
        print(
            f"[{label}] 随机参考：pooled acc={reference['pooled_accuracy']:.4f} "
            f"逐折 mean={reference['accuracy_mean']:.4f} ± {reference['accuracy_std']:.4f} | "
            f"pooled diff={reference['pooled_accuracy'] - pooled['accuracy']:+.4f}"
        )
        print(
            f"[{label}] 配对（参考 − 留一，按组等权）：mean={paired['mean_diff']:+.4f} "
            f"95% CI [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] "
            f"Wilcoxon p={p_text} (n={paired['n_pairs']})"
        )
        if permutation is not None:
            print(
                f"[{label}] 置换：null={permutation['null_mean']:.4f} ± {permutation['null_std']:.4f} "
                f"p95={permutation['null_p95']:.4f} observed={permutation['observed']:.4f} "
                f"z={permutation['z_score']:.1f} p={permutation['p_value']:.4f}"
            )
        if mode == "day" and loo["n_groups"] <= 5:
            print(
                f"[{label}] 注意：只有 {loo['n_groups']} 个 day 簇，bootstrap CI 较粗；"
                f"逐 day 结果见 paired_{label}_{args.subject}.csv"
            )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
        save_classification_csv(
            out_dir / f"classification_report_{label}_{args.subject}.csv",
            args.subject,
            label,
            report,
        )
        save_paired_csv(
            out_dir / f"paired_{label}_{args.subject}.csv",
            args.subject,
            label,
            loo,
            reference,
            run_day,
        )

        results[mode] = {
            "leave_one_out": loo,
            "reference_random_cv": reference,
            "paired": paired,
            "permutation": permutation,
        }
        predictions[mode] = y_pred
        summary_rows.append(
            summarize_mode(
                args.subject,
                args.task,
                args.top_k,
                label,
                loo,
                reference,
                paired,
                permutation,
                feats.shape[1],
                len(y),
                args.model,
                args.bands_mode,
                float(1 / num_classes),
                folds,
            )
        )

    comparison = None
    if args.leave_out == "both":
        comparison = compare_run_vs_day(
            y,
            dataset.day_groups,
            list(dataset.day_ids),
            predictions["run"],
            predictions["day"],
            args.seed,
        )
        rows = [
            {
                "subject": args.subject,
                "day": row["day"],
                "n_test": row["n_test"],
                "loro_accuracy": row["loro_accuracy"],
                "lodo_accuracy": row["lodo_accuracy"],
                "diff_loro_minus_lodo": row["diff_loro_minus_lodo"],
            }
            for row in comparison["rows"]
        ]
        write_rows(
            out_dir / f"run_vs_day_{args.subject}.csv",
            rows,
            ["subject", "day", "n_test", "loro_accuracy", "lodo_accuracy", "diff_loro_minus_lodo"],
        )
        p_text = "-" if comparison["wilcoxon_p"] is None else f"{comparison['wilcoxon_p']:.4g}"
        print(
            f"\n[LORO − LODO] mean={comparison['mean_diff']:+.4f} "
            f"95% CI [{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] "
            f"Wilcoxon p={p_text} (n={len(comparison['rows'])})"
        )

    payload = {
        "subject": args.subject,
        "task": "loro_lodo_classification",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "classes": class_names,
        "runs": list(dataset.run_names),
        "days": list(dataset.day_names),
        "n_samples": int(len(y)),
        "n_features": int(feats.shape[1]),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": sfreq,
        "use_64_channels": args.use_64_channels,
        "leave_out": args.leave_out,
        "reference_cv_folds": folds,
        "modes": results,
        "run_vs_day": comparison,
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
    parser = argparse.ArgumentParser(description="leave-one-run-out / leave-one-day-out 句类分类（log 频带功率 + 线性分类器）。")
    parser.add_argument("--subject", nargs="+", default=["sub-01"], help="被试编号，可指定多个，如 --subject sub-01 sub-02")
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"), help="读取哪一组 CSV / run pkl")
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--sweep-bands", action="store_true", help="逐个频段扫描（扫描模式不做置换）")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器")
    parser.add_argument("--leave-out", default="both", choices=("run", "day", "both"), help="留出 run（LORO）/ day（LODO）/ 两者都做")
    parser.add_argument("--folds", type=int, default=5, help="随机参考 CV 的折数")
    parser.add_argument("--permutations", type=int, default=2000, help="每个留一模式的置换次数；LORO 每折 45 组、成本较高，LODO 只有 5 折。0 表示跳过")
    parser.add_argument("--seed", type=int, default=42, help="随机参考 CV、配对 bootstrap 与置换的种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = list(args.subject)
    base_folds = args.folds
    run_name = args.run_name or f"loro_lodo_{datetime.now():%Y%m%d_%H%M%S}"
    all_rows: list[dict] = []

    for subject in subjects:
        args.subject = subject
        args.folds = base_folds  # 上一个被试可能因样本太少降过折数，重置
        out_dir = args.out_dir / run_name
        if len(subjects) > 1:
            out_dir = out_dir / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== 被试 {subject} =====")
        all_rows.extend(run_subject(args, out_dir))

    if len(subjects) > 1:
        run_root = args.out_dir / run_name
        if args.sweep_bands:
            paths = sorted(run_root.glob("*/sweep_*.csv"))
            if paths:
                sweep_path = run_root / "sweep_all.csv"
                combine_csvs(paths, sweep_path)
                print(f"[Saved] {sweep_path}")
        elif all_rows:
            summary_path = run_root / "summary_all.csv"
            write_rows(summary_path, all_rows, SUMMARY_FIELDS)
            print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
