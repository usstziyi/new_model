"""实验 03：run 内中心化 / 标准化，检验类别信息是否被 run 级状态漂移淹没。

思路
----
同一 run 内记录条件稳定，跨 run 的电极阻抗、注意力、漂移等慢变化往往远大于
语义相关的微弱差异。对每个 run 的特征减去该 run 的均值（只用输入、不用标签），
可以消掉 run 级共模成分，保留“同一 run 内不同类别之间的相对差异”。

在基线特征（频带功率 / Hjorth）上比较三种归一化：
``none``（同 exp01）、``runcenter``（减 run 均值）、``runz``（run 内 z-score）。
仍然按 run 分组做 5 折，验证集只用于选配置。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402
from plan_auto.common.utils import Tee, set_seed, timestamp  # noqa: E402

FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/features"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp03_runcenter/outputs"


def load_features(subject: str) -> dict[str, np.ndarray]:
    path = FEATURE_ROOT / subject / "features.npz"
    with np.load(path) as cached:
        return {key: cached[key] for key in cached.files}


def normalize_features(x: np.ndarray, runs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    out = x.astype(np.float32, copy=True)
    for run in np.unique(runs):
        mask = runs == run
        block = out[mask]
        mean = block.mean(axis=0, keepdims=True)
        block -= mean
        if mode == "runz":
            std = block.std(axis=0, keepdims=True)
            block /= np.maximum(std, 1e-6)
        out[mask] = block
    return out


def make_model(kind: str, n_features: int):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(128, n_features, 256), svd_solver="randomized", random_state=0),
        LogisticRegression(C=0.1, max_iter=2000),
    )


def run_subject(subject: str, seed: int = 42, n_splits: int = 5) -> dict:
    set_seed(seed)
    cache = D.load_cache(subject)
    labels = np.asarray(cache["labels"])
    runs = np.asarray(cache["runs"])
    n_chars = np.asarray(cache["n_chars"])
    raw = load_features(subject)
    base = {
        "bp": raw["bp"],
        "bpwin+hjorth": np.concatenate([raw["bpwin"], raw["hjorth"]], axis=1),
    }

    variants: dict[str, np.ndarray] = {}
    for name, matrix in base.items():
        for mode in ("none", "runcenter", "runz"):
            variants[f"{name}|{mode}"] = normalize_features(matrix, runs, mode)

    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    matched_mask = np.zeros(len(labels), dtype=bool)
    matched_mask[matched] = True

    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    n_classes = int(labels.max()) + 1
    y_pred = np.full(len(labels), -1, dtype=np.int64)
    prob = np.zeros((len(labels), n_classes), dtype=np.float32)
    rows, val_scores = [], defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(folds):
        fit_idx, val_idx = D.split_train_val(train_idx, runs, labels, val_ratio=0.25, seed=seed + fold)
        scores, fitted = {}, {}
        for key, matrix in variants.items():
            name, mode = key.split("|")
            kinds = ("lr", "pca_lr") if name == "bp" else ("pca_lr",)
            for kind in kinds:
                config = f"{key}|{kind}"
                model = make_model(kind, matrix.shape[1])
                model.fit(matrix[fit_idx], labels[fit_idx])
                score = float(np.mean(model.predict(matrix[val_idx]) == labels[val_idx]))
                scores[config] = score
                fitted[config] = model
                val_scores[f"{mode}|{kind}"].append(score)

        best = max(scores, key=lambda k: (scores[k], -list(scores).index(k)))
        model = fitted[best]
        matrix = variants[best.split("|")[0] + "|" + best.split("|")[1]]
        pred = model.predict(matrix[test_idx])
        y_pred[test_idx] = pred
        prob[test_idx] = model.predict_proba(matrix[test_idx])
        rows.append(
            {
                "fold": fold,
                "selected": best,
                "val_acc": scores[best],
                "test_acc": float(np.mean(pred == labels[test_idx])),
                "test_matched_acc": float(
                    np.mean(pred[matched_mask[test_idx]] == labels[test_idx][matched_mask[test_idx]])
                ),
            }
        )
        print(f"[{subject}] fold {fold}: {best} val={scores[best]:.3f} "
              f"test={rows[-1]['test_acc']:.3f} matched={rows[-1]['test_matched_acc']:.3f}", flush=True)

    result = M.classification_metrics(labels, y_pred, n_classes)
    _, low, high = M.cluster_bootstrap_ci(labels, y_pred, runs)
    result.update(
        {
            "subject": subject,
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "matched_accuracy": float(np.mean(y_pred[matched_mask] == labels[matched_mask])),
            "matched_n": int(matched_mask.sum()),
            "binomial_p_matched": M.binomial_p(
                int(np.sum(y_pred[matched_mask] == labels[matched_mask])),
                int(matched_mask.sum()),
                1 / n_classes,
            ),
            "mean_val_acc_by_variant": {k: float(np.mean(v)) for k, v in val_scores.items()},
        }
    )
    out_dir = OUT_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "folds.csv", index=False)
    np.savez_compressed(out_dir / "oof.npz", y_true=labels, y_pred=y_pred, prob=prob,
                        runs=runs, matched_mask=matched_mask)
    M.save_json(out_dir / "summary.json", result)
    print(f"[{subject}] acc={result['accuracy']:.3f} CI=[{low:.3f},{high:.3f}] "
          f"matched={result['matched_accuracy']:.3f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Tee(OUT_ROOT / f"exp03_{timestamp()}.log"):
        results = [run_subject(s, seed=args.seed) for s in args.subjects]
        pd.DataFrame(results).to_csv(OUT_ROOT / "results.csv", index=False)


if __name__ == "__main__":
    main()
