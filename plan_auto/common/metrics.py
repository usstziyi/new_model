"""统一的分类评估与显著性检验工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, n_classes: int | None = None) -> dict:
    """返回 acc / balanced acc / macro-F1 / 每类召回率 / 混淆矩阵。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = cm.sum(axis=1)
    return {
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_recall": [float(x) for x in per_class],
        "support": [int(x) for x in support],
        "confusion_matrix": cm.tolist(),
        "chance_accuracy": 1.0 / n_classes,
    }


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "accuracy",
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """对试次做有放回重采样，返回指标的 95% 置信区间。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "accuracy":
            values[i] = np.mean(y_true[idx] == y_pred[idx])
        elif metric == "balanced_accuracy":
            values[i] = balanced_accuracy_score(y_true[idx], y_pred[idx])
        elif metric == "macro_f1":
            values[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        else:
            raise ValueError(f"未知 metric: {metric}")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def binomial_p(n_correct: int, n_total: int, chance: float, alternative: str = "greater") -> float:
    """相对机会水平的单侧二项检验 p 值。"""
    return float(
        binomtest(int(n_correct), int(n_total), float(chance), alternative=alternative).pvalue
    )


def mcnemar_p(correct_a, correct_b) -> float:
    """McNemar 检验：比较两个模型在同一批试次上的正确/错误模式。"""
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def permutation_p(
    y_true,
    y_pred,
    groups,
    metric_fn,
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """在组内置换标签，得到“模型预测与标签无关联”的零分布 p 值。

    对每个 run 内部打乱标签，保持每个 run 的标签构成不变，只破坏试次与标签的
    对应关系。适合评估“预测是否真的依赖单试次信息”。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    observed = metric_fn(y_true, y_pred)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = y_true.copy()
        for group in np.unique(groups):
            mask = groups == group
            shuffled[mask] = rng.permutation(y_true[mask])
        null[i] = metric_fn(shuffled, y_pred)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def cluster_bootstrap_ci(
    y_true,
    y_pred,
    groups,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """按组（run）重采样，返回 (点估计, 2.5%, 97.5%)。

    同一 run 内的试次高度相关，按试次重采样会低估不确定性；这里以 run 为
    独立单位重采样整簇，得到更保守的置信区间。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    observed = float(np.mean(y_true == y_pred))
    unique, inverse = np.unique(groups, return_inverse=True)
    correct_per_group = np.bincount(inverse, weights=(y_true == y_pred).astype(float), minlength=len(unique))
    total_per_group = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        values[i] = correct_per_group[picked].sum() / max(total_per_group[picked].sum(), 1)
    return observed, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def save_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
