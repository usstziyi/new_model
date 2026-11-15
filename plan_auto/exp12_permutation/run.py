"""实验 12：训练级置换检验。

此前的置换只打乱已训练模型的 OOF 预测对应关系；审稿人更认可“换标签重新训练”
的零分布。这里对每个被试重复：
1. 在每个 run 内随机打乱类别标签（保持每个 run 的类别构成与试次数）；
2. 用与主实验相同的按 run 分组 5 折 CV 重新训练 LDA；
3. 记录 accuracy 与 run 内类别平衡召回。
最后给出经验 p 值：``(1 + #{null >= observed}) / (n_perm + 1)``。

同时跑 raw 与位置去趋势两套特征，验证混杂控制后的结果确实落在零分布内。
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402

HIGHPASS = 1.5
FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/features_hp15"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp12_permutation/outputs"


def normalized_position(runs: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(runs), dtype=np.float64)
    for run in np.unique(runs):
        idx = np.flatnonzero(runs == run)
        pos[idx] = np.arange(len(idx)) / max(len(idx) - 1, 1)
    return pos


def position_detrend(x: np.ndarray, runs: np.ndarray, pos: np.ndarray, degree: int = 3) -> np.ndarray:
    out = x.astype(np.float64, copy=True)
    for run in np.unique(runs):
        mask = runs == run
        p = pos[mask]
        if len(p) < degree + 2:
            continue
        design = np.vstack([p ** d for d in range(degree + 1)]).T
        block = out[mask]
        coef, *_ = np.linalg.lstsq(design, block, rcond=None)
        out[mask] = block - design @ coef
    return out.astype(np.float32)


def run_balanced_recall(y: np.ndarray, pred: np.ndarray, runs: np.ndarray) -> float:
    recalls, weights = [], []
    for run in np.unique(runs):
        mask = runs == run
        yy, pp = y[mask], pred[mask]
        classes = np.unique(yy)
        if len(classes) < 2:
            continue
        recalls.append(np.mean([np.mean(pp[yy == c] == c) for c in classes]))
        weights.append(len(yy))
    return float(np.average(recalls, weights=weights))


def _lda():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def cv_predict(x, y, folds):
    pred = np.full(len(y), -1, dtype=np.int64)
    for tr, te in folds:
        model = _lda()
        model.fit(x[tr], y[tr])
        pred[te] = model.predict(x[te])
    return pred


def permute_within_run(y: np.ndarray, runs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = y.copy()
    for run in np.unique(runs):
        mask = runs == run
        out[mask] = rng.permutation(y[mask])
    return out


def run_subject(task):
    subject, n_perm, seed, n_splits = task
    D.build_cache(subject, highpass=HIGHPASS)
    cache = D.load_cache(subject, highpass=HIGHPASS)
    labels, runs = cache["labels"], cache["runs"]
    with np.load(FEATURE_ROOT / subject / "features.npz") as cached:
        bp = cached["bp"].copy()
    pos = normalized_position(runs)
    folds = D.grouped_folds(runs, labels, n_splits, seed)
    variants = {"raw": bp, "posdetrend": position_detrend(bp, runs, pos)}

    results = {}
    for name, x in variants.items():
        pred = cv_predict(x, labels, folds)
        observed = {
            "accuracy": float(np.mean(pred == labels)),
            "run_balanced_recall": run_balanced_recall(labels, pred, runs),
        }
        rng = np.random.default_rng(seed + (0 if name == "raw" else 10_000))
        null_acc = np.empty(n_perm)
        null_rb = np.empty(n_perm)
        for i in range(n_perm):
            y_perm = permute_within_run(labels, runs, rng)
            p = cv_predict(x, y_perm, folds)
            null_acc[i] = np.mean(p == y_perm)
            null_rb[i] = run_balanced_recall(y_perm, p, runs)
            if (i + 1) % 20 == 0:
                print(f"[{subject}][{name}] perm {i + 1}/{n_perm}", flush=True)
        results[name] = {
            "observed": observed,
            "null_accuracy": null_acc,
            "null_run_balanced_recall": null_rb,
            "p_accuracy": float((1 + np.sum(null_acc >= observed["accuracy"])) / (n_perm + 1)),
            "p_run_balanced_recall": float(
                (1 + np.sum(null_rb >= observed["run_balanced_recall"])) / (n_perm + 1)
            ),
        }
        print(
            f"[{subject}][{name}] obs_acc={observed['accuracy']:.3f} "
            f"p_acc={results[name]['p_accuracy']:.4f} | "
            f"obs_rb={observed['run_balanced_recall']:.3f} "
            f"p_rb={results[name]['p_run_balanced_recall']:.4f}",
            flush=True,
        )

    out = OUT_ROOT / subject
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "permutation.npz",
        labels=labels,
        runs=runs,
        **{
            f"{name}_{key}": value
            for name, payload in results.items()
            for key, value in payload.items()
        },
    )
    summary = {name: {"observed": payload["observed"],
                      "p_accuracy": payload["p_accuracy"],
                      "p_run_balanced_recall": payload["p_run_balanced_recall"]}
               for name, payload in results.items()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"subject": subject, **{f"{name}_{k}": v for name, payload in results.items()
                                   for k, v in payload["observed"].items()},
            **{f"{name}_{k}": v for name, payload in results.items()
               for k, v in (("p_accuracy", payload["p_accuracy"]),
                            ("p_run_balanced_recall", payload["p_run_balanced_recall"]))},
            **{f"{name}_null_acc_mean": float(np.mean(payload["null_accuracy"])) for name, payload in results.items()},
            **{f"{name}_null_rb_mean": float(np.mean(payload["null_run_balanced_recall"])) for name, payload in results.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, dest="n_splits")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.n_perm, args.seed, args.n_splits) for s in args.subjects]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(run_subject, tasks))
    table = pd.DataFrame(rows).sort_values("subject")
    table.to_csv(OUT_ROOT / "results.csv", index=False)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
