"""实验 05：多特征 / 多模型概率集成（不挑单一配置，降低选择方差）。

exp01 每折只保留验证集上最好的一个配置，方差较大。这里固定一组互补的
特征 + 线性模型，把它们的测试概率平均（权重可由验证集准确率决定），
看集成是否能稳定超过单模型。
"""

from __future__ import annotations

import argparse
import sys
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

FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/features"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp05_ensemble/outputs"


def load_features(subject: str) -> dict[str, np.ndarray]:
    path = FEATURE_ROOT / subject / "features.npz"
    with np.load(path) as cached:
        return {key: cached[key] for key in cached.files}


def build_members() -> dict:
    return {
        "bp_lr": ("bp", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
        "bp_pca": (
            "bp",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=128, svd_solver="randomized", random_state=0),
                LogisticRegression(C=0.1, max_iter=2000),
            ),
        ),
        "bpwin_pca": (
            "bpwin",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=256, svd_solver="randomized", random_state=0),
                LogisticRegression(C=0.1, max_iter=2000),
            ),
        ),
        "hjorth_lda": (
            "hjorth",
            make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ),
        "erp_pca": (
            "erp",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=128, svd_solver="randomized", random_state=0),
                LogisticRegression(C=0.1, max_iter=2000),
            ),
        ),
    }


def run_subject(subject: str, seed: int = 42, n_splits: int = 5, weighted: bool = True) -> dict:
    set_seed(seed)
    cache = D.load_cache(subject)
    labels = np.asarray(cache["labels"])
    runs = np.asarray(cache["runs"])
    n_chars = np.asarray(cache["n_chars"])
    raw = load_features(subject)
    members = build_members()

    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    matched_mask = np.zeros(len(labels), dtype=bool)
    matched_mask[matched] = True

    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    n_classes = int(labels.max()) + 1
    ensemble_prob = np.zeros((len(labels), n_classes), dtype=np.float32)
    member_prob = {name: np.zeros((len(labels), n_classes), dtype=np.float32) for name in members}
    rows = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        fit_idx, val_idx = D.split_train_val(train_idx, runs, labels, val_ratio=0.25, seed=seed + fold)
        val_acc, test_probs, fitted = {}, {}, {}
        for name, (feature_key, model) in members.items():
            matrix = raw[feature_key]
            model.fit(matrix[fit_idx], labels[fit_idx])
            fitted[name] = model
            val_acc[name] = float(np.mean(model.predict(matrix[val_idx]) == labels[val_idx]))
            prob = model.predict_proba(matrix[test_idx])
            test_probs[name] = prob
            member_prob[name][test_idx] = prob

        if weighted:
            scores = np.array([val_acc[name] for name in members])
            weights = np.exp((scores - scores.max()) / 0.05)
            weights /= weights.sum()
        else:
            weights = np.full(len(members), 1.0 / len(members))
        blend = sum(weights[i] * test_probs[name] for i, name in enumerate(members))
        ensemble_prob[test_idx] = blend
        rows.append(
            {
                "fold": fold,
                **{f"val_{name}": val_acc[name] for name in members},
                **{f"w_{name}": weights[i] for i, name in enumerate(members)},
                "test_acc": float(np.mean(blend.argmax(1) == labels[test_idx])),
            }
        )
        print(
            f"[{subject}] fold {fold} ensemble={rows[-1]['test_acc']:.3f} "
            f"members=" + " ".join(f"{n}:{val_acc[n]:.3f}" for n in members),
            flush=True,
        )

    y_pred = ensemble_prob.argmax(1)
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
            "member_matched_accuracy": {
                name: float(np.mean(member_prob[name][matched_mask].argmax(1) == labels[matched_mask]))
                for name in members
            },
        }
    )
    out_dir = OUT_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "folds.csv", index=False)
    np.savez_compressed(
        out_dir / "oof.npz",
        y_true=labels,
        y_pred=y_pred,
        prob=ensemble_prob,
        runs=runs,
        matched_mask=matched_mask,
        **{f"prob_{name}": member_prob[name] for name in members},
    )
    M.save_json(out_dir / "summary.json", result)
    print(f"[{subject}] acc={result['accuracy']:.3f} CI=[{low:.3f},{high:.3f}] "
          f"matched={result['matched_accuracy']:.3f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-weight", action="store_true")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Tee(OUT_ROOT / f"exp05_{timestamp()}.log"):
        results = [run_subject(s, seed=args.seed, weighted=not args.no_weight) for s in args.subjects]
        pd.DataFrame(results).to_csv(OUT_ROOT / "results.csv", index=False)


if __name__ == "__main__":
    main()
