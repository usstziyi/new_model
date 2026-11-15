"""实验 09：混杂控制——把“试次在 run 内的位置”从特征里剔除后还能不能解码。

背景（审计发现）
----------------
同一 run 内语义类别按时间块顺序呈现：sub-01 的类别平均位置从 0.04 单调升到
0.96，仅用“归一化试次位置”做分类，按 run 分组的 10 类准确率就有 36.8%。
这意味着任何随 run 缓慢变化的分量（电极漂移、阻抗、疲劳、带通里的低频功率）
都会伪装成“语义解码”。

本实验做两件事：
1. 用 1.5 Hz 高通缓存重建特征，物理上削掉慢漂移；
2. 在特征层按 run 内位置做三次多项式去趋势（position detrend），把能被位置
   解释的成分减掉，只保留同位置附近的试次差异；
3. 评估指标改成 **run 内类别平衡召回**（每个 run 内各类 recall 取平均，
   再按 run 加权），消除 run 内类别比例不均带来的虚高。

如果位置去趋势后仍显著高于 10%，才说明存在独立于时间/漂移的语义信息。
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402

HIGHPASS = 1.5
FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/features_hp15"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp09_confound/outputs"
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("lowgamma", 30, 45))


def _psd(x, sfreq, nperseg):
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    out = np.empty((x.shape[0], x.shape[1], len(BANDS)), dtype=np.float32)
    for i, (_, low, high) in enumerate(BANDS):
        mask = (freqs >= low) & (freqs < high)
        out[:, :, i] = np.log10(np.maximum(psd[:, :, mask].mean(-1), 1e-12))
    return out


def _hjorth(eeg):
    x = np.asarray(eeg, dtype=np.float32)
    d1, d2 = np.diff(x, axis=-1), np.diff(x, n=2, axis=-1)
    v0, v1, v2 = x.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(np.divide(v1, np.maximum(v0, 1e-12)))
    mob2 = np.sqrt(np.divide(v2, np.maximum(v1, 1e-12)))
    return np.stack([np.log10(np.maximum(v0, 1e-12)), mob, np.divide(mob2, np.maximum(mob, 1e-12))], -1).reshape(len(x), -1)


def build_features(subject: str) -> dict:
    out = FEATURE_ROOT / subject
    path = out / "features.npz"
    if path.exists():
        with np.load(path) as c:
            return {k: c[k] for k in c.files}
    D.build_cache(subject, highpass=HIGHPASS)
    cache = D.load_cache(subject, highpass=HIGHPASS)
    eeg = np.asarray(cache["eeg"], dtype=np.float32)
    feats = {
        "bp": _psd(eeg, 250.0, 256).reshape(len(eeg), -1),
        "hjorth": _hjorth(eeg),
        "erp": eeg[:, :, ::12].reshape(len(eeg), -1),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feats)
    print(f"[Features] {subject} {[(k, v.shape) for k, v in feats.items()]}", flush=True)
    return feats


def normalized_position(runs: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(runs), dtype=np.float64)
    for run in np.unique(runs):
        idx = np.flatnonzero(runs == run)
        pos[idx] = np.arange(len(idx)) / max(len(idx) - 1, 1)
    return pos


def position_detrend(x: np.ndarray, runs: np.ndarray, pos: np.ndarray, degree: int = 3) -> np.ndarray:
    """按 run 内位置做多项式去趋势，逐特征维度。"""
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
    """每个 run 内各类 recall 平均，再按 run 试次数加权。"""
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


def _lda(pca_dim: int = 0):
    steps = [StandardScaler()]
    if pca_dim:
        steps.append(PCA(n_components=pca_dim, svd_solver="randomized", random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return make_pipeline(*steps)


def _models(feats: dict, n: int, n_classes: int):
    return {
        "bp_lda": (feats["bp"], lambda: _lda()),
        "band_ens": (feats["bp"].reshape(n, 122, 5), lambda: None),  # 特殊处理
        "hjorth_lda": (feats["hjorth"], lambda: _lda()),
        "erp_pca": (feats["erp"], lambda: _lda(96)),
    }


def run_subject(task):
    subject, seed, n_splits = task
    feats = build_features(subject)
    cache = D.load_cache(subject, highpass=HIGHPASS)
    labels, runs, n_chars = cache["labels"], cache["runs"], cache["n_chars"]
    n = len(labels)
    n_classes = int(labels.max()) + 1
    pos = normalized_position(runs)

    variants = {
        "raw": {k: v for k, v in feats.items()},
        "posdetrend": {k: position_detrend(v, runs, pos) for k, v in feats.items()},
    }
    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    mask = np.zeros(n, dtype=bool)
    mask[matched] = True

    # 位置基线
    design = np.column_stack([pos ** d for d in range(1, 5)])
    pos_pred = np.full(n, -1)
    for tr, te in folds:
        model = _lda()
        model.fit(design[tr], labels[tr])
        pos_pred[te] = model.predict(design[te])

    results = []

    def record(name, pred):
        results.append(
            {
                "subject": subject,
                "setting": name,
                "accuracy": float(np.mean(pred == labels)),
                "run_balanced_recall": run_balanced_recall(labels, pred, runs),
                "matched_accuracy": float(np.mean(pred[mask] == labels[mask])),
                "chance": 1 / n_classes,
            }
        )
        print(f"[{subject}] {name}: acc={results[-1]['accuracy']:.3f} "
              f"runbal={results[-1]['run_balanced_recall']:.3f} "
              f"matched={results[-1]['matched_accuracy']:.3f}", flush=True)

    record("position_only", pos_pred)

    for variant, matrices in variants.items():
        preds = {k: np.full(n, -1) for k in ("bp_lda", "band_ens", "hjorth_lda", "erp_pca")}
        for tr, te in folds:
            y = labels[tr]
            for name in ("bp_lda", "hjorth_lda", "erp_pca"):
                x = matrices[{"bp_lda": "bp", "hjorth_lda": "hjorth", "erp_pca": "erp"}[name]]
                pca_dim = 96 if name == "erp_pca" else 0
                model = _lda(pca_dim)
                model.fit(x[tr], y)
                preds[name][te] = model.predict(x[te])
            bp = matrices["bp"].reshape(n, 122, 5)
            band_prob = np.zeros((len(te), n_classes))
            for b in range(5):
                model = _lda()
                model.fit(bp[tr, :, b], y)
                band_prob += model.predict_proba(bp[te, :, b])
            preds["band_ens"][te] = band_prob.argmax(1)
        for name, pred in preds.items():
            record(f"{variant}|{name}", pred)

    out = OUT_ROOT / subject
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    return {"subject": subject, "rows": results}


def pool(results):
    table = pd.DataFrame([row for item in results for row in item["rows"]])
    table.to_csv(OUT_ROOT / "all_results.csv", index=False)
    summary = (
        table.groupby("setting")[["accuracy", "run_balanced_recall", "matched_accuracy"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(OUT_ROOT / "pooled_results.csv")
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.seed, args.folds) for s in args.subjects]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_subject, tasks))
    pool(results)


if __name__ == "__main__":
    main()
