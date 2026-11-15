"""实验 01：手工特征 + 线性模型，按 run 分组交叉验证。

目的：在进入深度模型之前，先用可解释、训练成本低的特征确认数据里是否
存在**跨 run 可复现**的类别信息，并给出句长配平后的对照。

协议
----
* 5 折按 run 分组（同一 run 不进训练和测试两侧）。
* 每折再从训练 run 里留 25% 作验证集，用验证准确率选择特征/模型配置，
  然后用该配置在测试折上预测，避免“用测试集挑配置”。
* 报告全量测试折准确率、句长配平子集准确率、按 run 的 bootstrap 置信区间。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402
from plan_auto.common.utils import Tee, set_seed, timestamp  # noqa: E402
from plan_auto.exp01_features import features as F  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "plan_auto/exp01_features/outputs"

# 64 导子集（与项目既有实验保持一致，用于协方差切空间）
CHANNELS_64 = [
    "FP1", "FPz", "Fp2", "AF3", "AFz", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "FTT9h", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TTP7h", "TP7", "CPP5h", "CP3", "CP1", "CPPz", "CP2", "CP4", "CPP6h", "TP8", "TTP8h",
    "P7", "P5", "P3", "POz", "P4", "P6", "P8",
    "PPO7", "PO3", "POO3", "Oz", "POO4", "PO4", "PPO8",
    "OI1", "Iz", "OI2",
]


def channel_indices_64() -> np.ndarray:
    import csv

    path = PROJECT_ROOT / "scripts/channels_122.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        names = [row["name"] for row in csv.DictReader(handle)]
    lookup = {name.lower(): i for i, name in enumerate(names)}
    return np.asarray([lookup[name.lower()] for name in CHANNELS_64], dtype=np.int64)


def make_model(kind: str, n_features: int):
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
        )
    if kind == "lr1":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        )
    if kind == "pca_lr":
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=min(128, n_features, 256), svd_solver="randomized", random_state=0),
            LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
        )
    if kind == "lda":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    raise ValueError(kind)


CONFIGS = (
    ("bp", "lr"),
    ("bp", "pca_lr"),
    ("bpwin", "pca_lr"),
    ("erp", "pca_lr"),
    ("hjorth", "lr"),
    ("hjorth", "lda"),
    ("tangent", "pca_lr"),
    ("bp+tangent", "pca_lr"),
    ("bpwin+hjorth", "pca_lr"),
)


def assemble_sets(raw: dict, eeg64: np.ndarray, fit_idx, val_idx, test_idx) -> dict:
    """返回 {配置名: (X_fit, X_val, X_test)}。"""
    sets = {}
    for name in ("bp", "bpwin", "erp", "hjorth"):
        arr = raw[name]
        sets[name] = (arr[fit_idx], arr[val_idx], arr[test_idx])

    tan_fit, rest = F.tangent_transform(eeg64[fit_idx], [eeg64[val_idx], eeg64[test_idx]])
    sets["tangent"] = (tan_fit, rest[0], rest[1])
    sets["bp+tangent"] = tuple(
        np.concatenate([raw["bp"][idx], tan], axis=1)
        for idx, tan in ((fit_idx, tan_fit), (val_idx, rest[0]), (test_idx, rest[1]))
    )
    sets["bpwin+hjorth"] = tuple(
        np.concatenate([raw["bpwin"][idx], raw["hjorth"][idx]], axis=1)
        for idx in (fit_idx, val_idx, test_idx)
    )
    return sets


def run_subject(subject: str, seed: int = 42, n_splits: int = 5) -> dict:
    set_seed(seed)
    cache = D.load_cache(subject)
    labels = np.asarray(cache["labels"])
    runs = np.asarray(cache["runs"])
    n_chars = np.asarray(cache["n_chars"])
    eeg = cache["eeg"]
    ch64 = channel_indices_64()

    raw = F.build_features(subject)
    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    matched_mask = np.zeros(len(labels), dtype=bool)
    matched_mask[matched] = True

    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    n = len(labels)
    n_classes = int(labels.max()) + 1
    y_pred = np.full(n, -1, dtype=np.int64)
    prob = np.zeros((n, n_classes), dtype=np.float32)
    y_val = np.full(n, -1, dtype=np.int64)
    val_prob = np.zeros((n, n_classes), dtype=np.float32)

    out_dir = OUT_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    val_scores: dict[str, list[float]] = defaultdict(list)
    eeg64 = np.asarray(eeg[:, ch64], dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(folds):
        fit_idx, val_idx = D.split_train_val(train_idx, runs, labels, val_ratio=0.25, seed=seed + fold)
        sets = assemble_sets(raw, eeg64, fit_idx, val_idx, test_idx)

        scores = {}
        fitted = {}
        for name, kind in CONFIGS:
            x_fit, x_val, _ = sets[name]
            model = make_model(kind, x_fit.shape[1])
            model.fit(x_fit, labels[fit_idx])
            val_pred = model.predict(x_val)
            score = float(np.mean(val_pred == labels[val_idx]))
            scores[f"{name}|{kind}"] = score
            fitted[f"{name}|{kind}"] = model
            val_scores[f"{name}|{kind}"].append(score)

        best_key = max(scores, key=lambda key: (scores[key], -CONFIGS.index(tuple(key.split("|")))))
        model = fitted[best_key]
        name = best_key.split("|")[0]
        x_fit, x_val, x_test = sets[name]
        pred = model.predict(x_test)
        y_pred[test_idx] = pred
        prob[test_idx] = model.predict_proba(x_test)
        y_val[val_idx] = model.predict(x_val)
        val_prob[val_idx] = model.predict_proba(x_val)

        test_acc = float(np.mean(pred == labels[test_idx]))
        selection_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_fit": len(fit_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "selected": best_key,
                "val_acc": scores[best_key],
                "test_acc": test_acc,
                "test_matched_acc": float(
                    np.mean(pred[matched_mask[test_idx]] == labels[test_idx][matched_mask[test_idx]])
                ),
                "best_possible_val": max(scores.values()),
            }
        )
        print(
            f"[{subject}] fold {fold}: selected={best_key} val={scores[best_key]:.3f} "
            f"test={test_acc:.3f}",
            flush=True,
        )
        del sets, fitted
        import gc

        gc.collect()

    if np.any(y_pred < 0):
        raise RuntimeError("有试次没有预测结果")

    result = M.classification_metrics(labels, y_pred, n_classes)
    point, low, high = M.cluster_bootstrap_ci(labels, y_pred, runs)
    matched_acc = float(np.mean(y_pred[matched_mask] == labels[matched_mask]))
    result.update(
        {
            "subject": subject,
            "method": "nested_feature_selection",
            "accuracy": float(np.mean(y_pred == labels)),
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "matched_accuracy": matched_acc,
            "matched_n": int(matched_mask.sum()),
            "chance_accuracy": 1.0 / n_classes,
            "binomial_p_full": M.binomial_p(int(np.sum(y_pred == labels)), n, 1.0 / n_classes),
            "binomial_p_matched": M.binomial_p(
                int(np.sum(y_pred[matched_mask] == labels[matched_mask])), int(matched_mask.sum()), 1.0 / n_classes
            ),
            "mean_val_acc_by_config": {k: float(np.mean(v)) for k, v in val_scores.items()},
            "selected_counts": pd.Series([row["selected"] for row in selection_rows]).value_counts().to_dict(),
            "n_trials": int(n),
            "n_runs": int(np.unique(runs).size),
        }
    )

    np.savez_compressed(
        out_dir / "oof.npz",
        y_true=labels,
        y_pred=y_pred,
        prob=prob,
        y_val=y_val,
        val_prob=val_prob,
        runs=runs,
        n_chars=n_chars,
        matched_mask=matched_mask,
    )
    pd.DataFrame(selection_rows).to_csv(out_dir / "folds.csv", index=False)
    M.save_json(out_dir / "summary.json", result)
    print(
        f"[{subject}] OOF acc={result['accuracy']:.3f} "
        f"CI=[{low:.3f},{high:.3f}] matched={matched_acc:.3f} "
        f"(chance={1/n_classes:.3f})",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="实验01：特征 + 线性模型，按 run 分组 CV")
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    run_dir = OUT_ROOT
    run_dir.mkdir(parents=True, exist_ok=True)
    with Tee(run_dir / f"exp01_{timestamp()}.log"):
        results = []
        for subject in args.subjects:
            results.append(run_subject(subject, seed=args.seed, n_splits=args.n_splits))

        table = pd.DataFrame(results).sort_values("accuracy", ascending=False)
        table.to_csv(run_dir / "results.csv", index=False)

        # 跨被试汇总（同一套 10 类标签可直接拼）
        y_true, y_pred, groups, matched = [], [], [], []
        for subject in args.subjects:
            data = np.load(OUT_ROOT / subject / "oof.npz")
            y_true.append(data["y_true"])
            y_pred.append(data["y_pred"])
            groups.append(np.char.add(f"{subject}:", data["runs"].astype(str)))
            matched.append(data["matched_mask"])
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        groups = np.concatenate(groups)
        matched = np.concatenate(matched)
        pooled = M.classification_metrics(y_true, y_pred, len(D.CLASS_IDS))
        _, low, high = M.cluster_bootstrap_ci(y_true, y_pred, groups)
        pooled.update(
            {
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "matched_accuracy": float(np.mean(y_pred[matched] == y_true[matched])),
                "matched_n": int(matched.sum()),
                "binomial_p_full": M.binomial_p(int(np.sum(y_pred == y_true)), len(y_true), 1 / len(D.CLASS_IDS)),
                "binomial_p_matched": M.binomial_p(
                    int(np.sum(y_pred[matched] == y_true[matched])), int(matched.sum()), 1 / len(D.CLASS_IDS)
                ),
                "n_subjects": len(args.subjects),
            }
        )
        M.save_json(OUT_ROOT / "pooled_summary.json", pooled)
        print(
            f"[Pooled] acc={pooled['accuracy']:.3f} CI=[{low:.3f},{high:.3f}] "
            f"matched={pooled['matched_accuracy']:.3f} n={len(y_true)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
