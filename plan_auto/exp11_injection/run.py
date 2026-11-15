"""实验 11：半合成信号注入，校准流程能检出多大的语义效应。

做法
----
对真实 EEG（1.5–45 Hz，250 Hz），按真实类标签注入**类别特异的 alpha 爆发**
（每类固定 30 个通道 + 8–12 Hz 频率 + 随机初相 + Hann 包络），幅度按
``SNR = 注入 RMS / 单试次信号 RMS`` 控制。然后跑与主实验完全相同的
按 run 分组 5 折 CV（频带功率 / Hjorth / ERP + LDA），同时报告
raw 与位置去趋势两套结果。

用途：如果真实数据在校正后只有机会水平，而注入曲线显示流程在某个 SNR
以上能稳定检出，就可以把“未检出”解释为“效应低于检测阈”，而不是“流程无效”。
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
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp11_injection/outputs"
SNRS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8)
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("lowgamma", 30, 45))


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


def inject(eeg: np.ndarray, labels: np.ndarray, snr: float, seed: int, sfreq: float = 250.0) -> np.ndarray:
    """按类注入 alpha 爆发；snr=0 时返回原数组的 float32 副本。"""
    out = np.asarray(eeg, dtype=np.float32).copy()
    if snr <= 0:
        return out
    n_trials, n_chans, n_times = out.shape
    t = np.arange(n_times) / sfreq
    envelope = np.hanning(n_times).astype(np.float32)
    rng = np.random.default_rng(seed)
    signal_std = float(out.std())
    for class_id in np.unique(labels):
        idx = np.flatnonzero(labels == class_id)
        chans = rng.choice(n_chans, size=30, replace=False)
        spatial = rng.normal(size=len(chans)).astype(np.float32)
        spatial /= np.linalg.norm(spatial) + 1e-9
        freq = rng.uniform(8.0, 12.0)
        phases = rng.uniform(0, 2 * np.pi, size=len(idx)).astype(np.float32)
        carrier = np.sin(2 * np.pi * freq * t[None, :] + phases[:, None]).astype(np.float32)
        wave = carrier * envelope[None, :]  # (n, T)
        block = out[np.ix_(idx, chans)]  # (n, 30, T) copy
        block += (snr * signal_std) * wave[:, None, :] * spatial[None, :, None]
        out[np.ix_(idx, chans)] = block
    return out


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


def _features(eeg):
    return {
        "bp": _psd(eeg, 250.0, 256).reshape(len(eeg), -1),
        "hjorth": _hjorth(eeg),
        "erp": np.ascontiguousarray(eeg[:, :, ::12], dtype=np.float32).reshape(len(eeg), -1),
    }


def _lda(pca_dim: int = 0):
    steps = [StandardScaler()]
    if pca_dim:
        steps.append(PCA(n_components=pca_dim, svd_solver="randomized", random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return make_pipeline(*steps)


def cv_predict(x: np.ndarray, y: np.ndarray, folds, pca_dim: int = 0) -> np.ndarray:
    pred = np.full(len(y), -1, dtype=np.int64)
    for tr, te in folds:
        model = _lda(pca_dim)
        model.fit(x[tr], y[tr])
        pred[te] = model.predict(x[te])
    return pred


def run_subject(task):
    subject, seed, n_splits = task
    D.build_cache(subject, highpass=HIGHPASS)
    cache = D.load_cache(subject, highpass=HIGHPASS)
    eeg = np.asarray(cache["eeg"], dtype=np.float32)
    labels, runs, n_chars = cache["labels"], cache["runs"], cache["n_chars"]
    pos = normalized_position(runs)
    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    mask = np.zeros(len(labels), dtype=bool)
    mask[matched] = True

    rows = []
    for snr in SNRS:
        injected = inject(eeg, labels, snr, seed=seed + int(snr * 1000))
        feats = _features(injected)
        for name, matrix in feats.items():
            pca_dim = 96 if name == "erp" else 0
            for variant, x in (
                ("raw", matrix),
                ("posdetrend", position_detrend(matrix, runs, pos)),
            ):
                pred = cv_predict(x, labels, folds, pca_dim)
                rows.append(
                    {
                        "subject": subject,
                        "snr": snr,
                        "feature": name,
                        "variant": variant,
                        "accuracy": float(np.mean(pred == labels)),
                        "run_balanced_recall": run_balanced_recall(labels, pred, runs),
                        "matched_accuracy": float(np.mean(pred[mask] == labels[mask])),
                        "chance": 0.1,
                    }
                )
        print(f"[{subject}] snr={snr:.2f} done", flush=True)
        del injected, feats

    out = OUT_ROOT / subject
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, dest="n_splits")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, args.seed, args.n_splits) for s in args.subjects]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_subject, tasks))
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(OUT_ROOT / "all_results.csv", index=False)

    curve = (
        table.groupby(["feature", "variant", "snr"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            runbal_mean=("run_balanced_recall", "mean"),
            runbal_sd=("run_balanced_recall", "std"),
            matched_mean=("matched_accuracy", "mean"),
        )
        .reset_index()
    )
    curve.to_csv(OUT_ROOT / "pooled_curve.csv", index=False)

    # 检测阈：run-balanced recall 首次超过 baseline + 2*sd
    thresholds = {}
    for (feature, variant), part in curve.groupby(["feature", "variant"]):
        baseline = float(part[part.snr == 0].runbal_mean.iloc[0])
        base_sd = float(part[part.snr == 0].runbal_sd.iloc[0] or 0.0)
        crossed = part[(part.snr > 0) & (part.runbal_mean > baseline + 2 * base_sd)]
        thresholds[f"{feature}|{variant}"] = float(crossed.snr.min()) if len(crossed) else None
    import json

    (OUT_ROOT / "summary.json").write_text(
        json.dumps({"snrs": list(SNRS), "detection_thresholds": thresholds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(curve.to_string(index=False), flush=True)
    print("检测阈:", json.dumps(thresholds, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
