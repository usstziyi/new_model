"""实验 06：特征 × 频带 × 时间窗 × 通道组的系统性扫描 + 概率集成。

比 exp01 更快、更全面：固定配置、不做嵌套选择，每个配置都给出按 run 分组
的 5 折 OOF 结果；最后把互补配置的概率平均，看集成能否稳定超过单模型。

配置族
------
* ``bp_lr`` / ``bp_pca``：全时段频带功率
* ``band_*``：逐频带（delta/theta/alpha/beta/lowgamma）单独建模，再平均
* ``win_*``：4 个时间窗分别建模，再平均（检验语义信息的时间分布）
* ``chan_*``：额 / 中央 / 颞 / 顶 / 枕 通道组分别建模，再平均
* ``hjorth_lda`` / ``erp_pca``：时域与非线性特征
* ``ensemble_all``：以上全部概率等权平均
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
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

FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/features"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp06_sweep/outputs"

BAND_NAMES = ("delta", "theta", "alpha", "beta", "lowgamma")
CHANNEL_GROUPS = {
    "frontal": ("fp", "af", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "fz"),
    "central": ("fc", "c1", "c2", "c3", "c4", "c5", "c6", "cz", "cp", "cpp"),
    "temporal": ("ft", "t7", "t8", "tp", "tt", "ftt", "ttp", "tp7", "tp8"),
    "parietal": ("p3", "p4", "p5", "p6", "p7", "p8", "pz", "poz"),
    "occipital": ("o1", "o2", "oz", "oi", "po3", "po4", "poo", "ppo"),
}


def _lr(n_features: int, pca: bool):
    if pca:
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=min(96, n_features, 256), svd_solver="randomized", random_state=0),
            LogisticRegression(C=0.1, max_iter=300, tol=1e-3),
        )
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300, tol=1e-3))


def _lda(n_features: int = 0, pca: bool = False):
    if pca:
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=min(96, n_features or 96), svd_solver="randomized", random_state=0),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def _load(subject: str) -> tuple[dict, dict]:
    cache = D.load_cache(subject)
    feats = {}
    with np.load(FEATURE_ROOT / subject / "features.npz") as cached:
        for key in ("bp", "bpwin", "hjorth", "erp"):
            feats[key] = cached[key]
    return cache, feats


def _channel_group_indices() -> dict[str, np.ndarray]:
    path = PROJECT_ROOT / "scripts/channels_122.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        names = [row["name"].lower() for row in csv.DictReader(handle)]
    groups: dict[str, list[int]] = {}
    for group, prefixes in CHANNEL_GROUPS.items():
        idx = [i for i, name in enumerate(names) if name.startswith(prefixes)]
        groups[group] = np.asarray(idx, dtype=np.int64)
    return groups


def evaluate_subject(args_tuple) -> dict:
    subject, seed, n_splits = args_tuple
    cache, feats = _load(subject)
    labels = np.asarray(cache["labels"])
    runs = np.asarray(cache["runs"])
    n_chars = np.asarray(cache["n_chars"])
    n_classes = int(labels.max()) + 1
    n = len(labels)
    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    matched_mask = np.zeros(n, dtype=bool)
    matched_mask[matched] = True

    # 拆出频带 / 时间窗 / 通道组视图
    bp = feats["bp"].reshape(n, 122, 5)
    bpwin = feats["bpwin"].reshape(n, 122, 4, 5)
    groups = _channel_group_indices()

    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    config_probs: dict[str, np.ndarray] = {}
    rows = []

    def ensure(name: str):
        if name not in config_probs:
            config_probs[name] = np.zeros((n, n_classes), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(folds):
        y_train = labels[train_idx]

        def fit_predict(name: str, x_train: np.ndarray, x_test: np.ndarray, model) -> None:
            ensure(name)
            model.fit(x_train, y_train)
            config_probs[name][test_idx] = model.predict_proba(x_test)

        # 全时段频带功率
        x = feats["bp"]
        fit_predict("bp_lr", x[train_idx], x[test_idx], _lr(x.shape[1], pca=False))
        fit_predict("bp_lda", x[train_idx], x[test_idx], _lda())
        fit_predict("bp_pca", x[train_idx], x[test_idx], _lr(x.shape[1], pca=True))

        # 逐频带
        for b, band in enumerate(BAND_NAMES):
            x = bp[:, :, b]
            fit_predict(f"band_{band}", x[train_idx], x[test_idx], _lda())

        # 逐时间窗
        for w in range(4):
            x = bpwin[:, :, w, :].reshape(n, -1)
            fit_predict(f"win_{w + 1}", x[train_idx], x[test_idx], _lda())

        # 通道组（5 个频带一起）
        for group, idx in groups.items():
            x = bp[:, idx, :].reshape(n, -1)
            fit_predict(f"chan_{group}", x[train_idx], x[test_idx], _lda(x.shape[1], pca=True))

        # 其它特征
        x = feats["hjorth"]
        fit_predict("hjorth_lda", x[train_idx], x[test_idx], _lda())
        x = feats["erp"]
        fit_predict("erp_pca", x[train_idx], x[test_idx], _lda(x.shape[1], pca=True))
        print(f"[{subject}] fold {fold} done", flush=True)

        # 频带平均 / 通道组平均
        for prefix in ("band_", "win_", "chan_"):
            names = [k for k in config_probs if k.startswith(prefix)]
            ensure(prefix.rstrip("_"))
            config_probs[prefix.rstrip("_")][test_idx] = np.mean(
                [config_probs[k][test_idx] for k in names], axis=0
            )
        # 全部平均
        names = [k for k in config_probs if not k.startswith(("band", "win", "chan")) or k in
                 ("band", "win", "chan")]
        ensure("ensemble_all")
        config_probs["ensemble_all"][test_idx] = np.mean([config_probs[k][test_idx] for k in names], axis=0)
        rows.append({"fold": fold, "n_test": len(test_idx)})

    results = []
    for name, prob in config_probs.items():
        y_pred = prob.argmax(axis=1)
        m = M.classification_metrics(labels, y_pred, n_classes)
        _, low, high = M.cluster_bootstrap_ci(labels, y_pred, runs)
        m.update(
            {
                "subject": subject,
                "config": name,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "matched_accuracy": float(np.mean(y_pred[matched_mask] == labels[matched_mask])),
                "matched_n": int(matched_mask.sum()),
            }
        )
        results.append(m)

    out_dir = OUT_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "oof.npz",
        y_true=labels,
        runs=runs,
        matched_mask=matched_mask,
        **{f"prob_{name}": prob for name, prob in config_probs.items()},
    )
    pd.DataFrame(results).to_csv(out_dir / "config_results.csv", index=False)
    best = max(results, key=lambda r: r["matched_accuracy"])
    print(
        f"[{subject}] best={best['config']} matched={best['matched_accuracy']:.3f} "
        f"acc={best['accuracy']:.3f} | ensemble matched="
        f"{next(r['matched_accuracy'] for r in results if r['config'] == 'ensemble_all'):.3f}",
        flush=True,
    )
    return {"subject": subject, "rows": results}


def pool(results: list[dict]) -> None:
    frames = []
    for item in results:
        frames.extend(item["rows"])
    table = pd.DataFrame(frames)
    table.to_csv(OUT_ROOT / "all_config_results.csv", index=False)

    pooled_rows = []
    for config in sorted(table.config.unique()):
        y_true, y_pred, groups, matched = [], [], [], []
        for subject in table.subject.unique():
            data = np.load(OUT_ROOT / subject / "oof.npz")
            prob = data[f"prob_{config}"]
            y_true.append(data["y_true"])
            y_pred.append(prob.argmax(1))
            groups.append(np.char.add(f"{subject}:", data["runs"].astype(str)))
            matched.append(data["matched_mask"])
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        groups = np.concatenate(groups)
        matched = np.concatenate(matched)
        m = M.classification_metrics(y_true, y_pred, len(D.CLASS_IDS))
        _, low, high = M.cluster_bootstrap_ci(y_true, y_pred, groups)
        m.update(
            {
                "config": config,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "matched_accuracy": float(np.mean(y_pred[matched] == y_true[matched])),
                "matched_n": int(matched.sum()),
                "matched_ci_low": None,
                "n_subjects": int(table.subject.nunique()),
            }
        )
        _, mlow, mhigh = M.cluster_bootstrap_ci(
            y_true[matched], y_pred[matched], groups[matched]
        )
        m["matched_ci_low"], m["matched_ci_high"] = mlow, mhigh
        pooled_rows.append(m)
    pooled_table = pd.DataFrame(pooled_rows).sort_values("matched_accuracy", ascending=False)
    pooled_table.to_csv(OUT_ROOT / "pooled_config_results.csv", index=False)
    print(pooled_table[["config", "accuracy", "matched_accuracy", "matched_ci_low",
                        "matched_ci_high", "accuracy_ci_low", "accuracy_ci_high"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.seed, args.folds) for s in args.subjects]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool_exec:
            results = list(pool_exec.map(evaluate_subject, tasks))
    else:
        results = [evaluate_subject(t) for t in tasks]
    pool(results)


if __name__ == "__main__":
    main()
