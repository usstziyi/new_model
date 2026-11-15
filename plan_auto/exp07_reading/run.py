"""实验 07：阅读阶段（task-read）的语义解码。

数据来源：``scripts/make_text_csv.py`` 生成的 ``sub-XX_reading.csv`` +
``scripts/export_top_classes.py --task reading`` 导出的
``datasets/top10_reading/<subject>/class_*.pkl``。

阅读阶段的 EEG 含更强的视觉/语言加工成分，作为想象阶段的对照：如果同一套
按 run 分组的流程在阅读阶段能明显高于机会水平，说明流程确实能读出语义信息，
而想象阶段的弱结果反映的是任务本身的信息量。

运行::

    uv run python plan_auto/exp07_reading/run.py --jobs 3
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402

TOP_ROOT = PROJECT_ROOT / "datasets/top10_reading"
CACHE_ROOT = PROJECT_ROOT / "plan_auto/cache/reading"
FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/reading_features"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp07_reading/outputs"

SFREQ_RAW = 500
N_CHANNELS = 122
FILTER = butter(4, (0.5, 45.0), btype="bandpass", fs=SFREQ_RAW, output="sos")
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("lowgamma", 30, 45))
CHANNEL_GROUPS = {
    "frontal": ("fp", "af", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "fz"),
    "central": ("fc", "c1", "c2", "c3", "c4", "c5", "c6", "cz", "cp", "cpp"),
    "temporal": ("ft", "t7", "t8", "tp", "tt", "ftt", "ttp", "tp7", "tp8"),
    "parietal": ("p3", "p4", "p5", "p6", "p7", "p8", "pz", "poz"),
    "occipital": ("o1", "o2", "oz", "oi", "po3", "po4", "poo", "ppo"),
}
_FILE = re.compile(r"class_(?P<cid>\d+)\.pkl$")


def cache_dir(subject: str) -> Path:
    return CACHE_ROOT / f"{subject}_sf250"


def build_cache(subject: str, overwrite: bool = False) -> Path:
    out = cache_dir(subject)
    if out.exists() and not overwrite:
        return out
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (p for p in (TOP_ROOT / subject).glob("class_*.pkl") if _FILE.search(p.name)),
        key=lambda p: int(_FILE.search(p.name)["cid"]),
    )
    if not paths:
        raise FileNotFoundError(f"{TOP_ROOT / subject} 下没有 class_*.pkl")

    arrays, labels, runs, days, n_chars, texts, class_ids, names = [], [], [], [], [], [], [], []
    for label, path in enumerate(paths):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        eeg = np.asarray(payload["eeg"], dtype=np.float64)
        if eeg.ndim == 4 and eeg.shape[1] == 1:
            eeg = eeg[:, 0]
        eeg = eeg[:, :N_CHANNELS] * 1e6
        eeg = sosfiltfilt(FILTER, eeg, axis=-1)
        eeg = np.ascontiguousarray(eeg[:, :, ::2], dtype=np.float16)
        meta = payload.get("meta", [{} for _ in range(len(eeg))])
        arrays.append(eeg)
        labels.append(np.full(len(eeg), label, dtype=np.int64))
        runs.append(np.asarray([int(m.get("run", -1)) for m in meta]))
        days.append(np.asarray([int(m.get("day", -1)) for m in meta]))
        n_chars.append(np.asarray([int(m.get("n_chars", -1)) for m in meta]))
        texts.extend(payload.get("text", [""] * len(eeg)))
        class_ids.append(int(payload["type_number"]))
        names.append(str(payload.get("type_name", "")))
        print(f"[Cache] {subject} class={payload['type_number']} n={len(eeg)} {eeg.shape}", flush=True)

    lengths = {a.shape[-1] for a in arrays}
    if len(lengths) != 1:
        raise RuntimeError(f"{subject} 长度不一致: {sorted(lengths)}")
    eeg = np.concatenate(arrays)
    np.save(out / "eeg.npy", eeg)
    np.save(out / "labels.npy", np.concatenate(labels))
    np.save(out / "runs.npy", np.concatenate(runs))
    np.save(out / "days.npy", np.concatenate(days))
    np.save(out / "n_chars.npy", np.concatenate(n_chars))
    (out / "texts.json").write_text(__import__("json").dumps(texts, ensure_ascii=False), encoding="utf-8")
    (out / "meta.json").write_text(
        __import__("json").dumps(
            {"subject": subject, "class_ids": class_ids, "class_names": names,
             "shape": list(eeg.shape), "sfreq": 250, "task": "reading"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[Cache] {subject} done {eeg.shape}", flush=True)
    return out


def load_cache(subject: str) -> dict:
    d = cache_dir(subject)
    return {
        "eeg": np.load(d / "eeg.npy", mmap_mode="r"),
        "labels": np.load(d / "labels.npy"),
        "runs": np.load(d / "runs.npy"),
        "n_chars": np.load(d / "n_chars.npy"),
    }


def _psd(x, sfreq, nperseg):
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    out = np.empty((x.shape[0], x.shape[1], len(BANDS)), dtype=np.float32)
    for i, (_, low, high) in enumerate(BANDS):
        mask = (freqs >= low) & (freqs < high)
        out[:, :, i] = np.log10(np.maximum(psd[:, :, mask].mean(-1), 1e-12))
    return out


def build_features(subject: str) -> dict:
    d = FEATURE_ROOT / subject
    path = d / "features.npz"
    if path.exists():
        with np.load(path) as c:
            return {k: c[k] for k in c.files}
    cache = load_cache(subject)
    eeg = np.asarray(cache["eeg"], dtype=np.float32)
    sfreq = 250.0
    n_windows = 4
    edges = np.linspace(0, eeg.shape[-1], n_windows + 1).astype(int)
    bpwin = np.concatenate(
        [_psd(eeg[..., edges[i]:edges[i + 1]], sfreq, 128) for i in range(n_windows)], axis=-1
    ).reshape(len(eeg), -1)
    feats = {
        "bp": _psd(eeg, sfreq, 256).reshape(len(eeg), -1),
        "bpwin": bpwin,
        "hjorth": _hjorth(eeg),
        "erp": eeg[:, :, ::int(round(sfreq / 20.0))].reshape(len(eeg), -1),
    }
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feats)
    print(f"[Features] {subject} {[(k, v.shape) for k, v in feats.items()]}", flush=True)
    return feats


def _hjorth(eeg):
    x = np.asarray(eeg, dtype=np.float32)
    d1, d2 = np.diff(x, axis=-1), np.diff(x, n=2, axis=-1)
    v0, v1, v2 = x.var(-1), d1.var(-1), d2.var(-1)
    mob = np.sqrt(np.divide(v1, np.maximum(v0, 1e-12)))
    mob2 = np.sqrt(np.divide(v2, np.maximum(v1, 1e-12)))
    return np.stack([np.log10(np.maximum(v0, 1e-12)), mob, np.divide(mob2, np.maximum(mob, 1e-12))], -1).reshape(len(x), -1)


def _lr(n_features, pca=False):
    if pca:
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=min(96, n_features), svd_solver="randomized", random_state=0),
            LogisticRegression(C=0.1, max_iter=300, tol=1e-3),
        )
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300, tol=1e-3))


def _lda(n_features=0, pca=False):
    if pca:
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=min(96, n_features or 96), svd_solver="randomized", random_state=0),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def channel_groups():
    with (PROJECT_ROOT / "scripts/channels_122.csv").open(encoding="utf-8-sig", newline="") as h:
        names = [row["name"].lower() for row in csv.DictReader(h)]
    return {g: np.asarray([i for i, n in enumerate(names) if n.startswith(p)], dtype=np.int64)
            for g, p in CHANNEL_GROUPS.items()}


def run_subject(task):
    subject, seed, n_splits = task
    build_cache(subject)
    feats = build_features(subject)
    cache = load_cache(subject)
    labels, runs, n_chars = cache["labels"], cache["runs"], cache["n_chars"]
    n = len(labels)
    n_classes = int(labels.max()) + 1
    matched = D.length_matched_indices(labels, n_chars, seed=seed)
    mask = np.zeros(n, dtype=bool)
    mask[matched] = True

    bp = feats["bp"].reshape(n, 122, 5)
    bpwin = feats["bpwin"].reshape(n, 122, 4, 5)
    groups = channel_groups()
    folds = D.grouped_folds(runs, labels, n_splits=n_splits, seed=seed)
    probs: dict[str, np.ndarray] = {}

    def ensure(name):
        probs.setdefault(name, np.zeros((n, n_classes), dtype=np.float32))

    for fold, (tr, te) in enumerate(folds):
        y = labels[tr]

        def fp(name, xtr, xte, model):
            ensure(name)
            model.fit(xtr, y)
            probs[name][te] = model.predict_proba(xte)

        x = feats["bp"]
        fp("bp_lr", x[tr], x[te], _lr(x.shape[1]))
        fp("bp_lda", x[tr], x[te], _lda())
        fp("bp_pca", x[tr], x[te], _lr(x.shape[1], pca=True))
        for b, (band, _, _) in enumerate(BANDS):
            x = bp[:, :, b]
            fp(f"band_{band}", x[tr], x[te], _lda())
        for w in range(4):
            x = bpwin[:, :, w, :].reshape(n, -1)
            fp(f"win_{w + 1}", x[tr], x[te], _lda())
        for g, idx in groups.items():
            x = bp[:, idx, :].reshape(n, -1)
            fp(f"chan_{g}", x[tr], x[te], _lda(x.shape[1], pca=True))
        x = feats["hjorth"]
        fp("hjorth_lda", x[tr], x[te], _lda())
        x = feats["erp"]
        fp("erp_pca", x[tr], x[te], _lda(x.shape[1], pca=True))

        for prefix in ("band_", "win_", "chan_"):
            names = [k for k in probs if k.startswith(prefix)]
            ensure(prefix.rstrip("_"))
            probs[prefix.rstrip("_")][te] = np.mean([probs[k][te] for k in names], axis=0)
        names = [k for k in probs if not k.startswith(("band", "win", "chan")) or k in ("band", "win", "chan")]
        ensure("ensemble_all")
        probs["ensemble_all"][te] = np.mean([probs[k][te] for k in names], axis=0)
        print(f"[{subject}] fold {fold} done", flush=True)

    rows = []
    for name, prob in probs.items():
        pred = prob.argmax(1)
        m = M.classification_metrics(labels, pred, n_classes)
        _, low, high = M.cluster_bootstrap_ci(labels, pred, runs)
        m.update({"subject": subject, "config": name, "accuracy_ci_low": low, "accuracy_ci_high": high,
                  "matched_accuracy": float(np.mean(pred[mask] == labels[mask])), "matched_n": int(mask.sum())})
        rows.append(m)
    out = OUT_ROOT / subject
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "config_results.csv", index=False)
    np.savez_compressed(out / "oof.npz", y_true=labels, runs=runs, matched_mask=mask,
                        **{f"prob_{k}": v for k, v in probs.items()})
    best = max(rows, key=lambda r: r["matched_accuracy"])
    print(f"[{subject}] best={best['config']} acc={best['accuracy']:.3f} matched={best['matched_accuracy']:.3f}", flush=True)
    return {"subject": subject, "rows": rows}


def pool(results):
    table = pd.DataFrame([row for item in results for row in item["rows"]])
    table.to_csv(OUT_ROOT / "all_config_results.csv", index=False)
    pooled = []
    for config in sorted(table.config.unique()):
        yt, yp, gr, mt = [], [], [], []
        for subject in table.subject.unique():
            d = np.load(OUT_ROOT / subject / "oof.npz")
            yt.append(d["y_true"])
            yp.append(d[f"prob_{config}"].argmax(1))
            gr.append(np.char.add(f"{subject}:", d["runs"].astype(str)))
            mt.append(d["matched_mask"])
        yt, yp, gr, mt = map(np.concatenate, (yt, yp, gr, mt))
        m = M.classification_metrics(yt, yp, 10)
        _, low, high = M.cluster_bootstrap_ci(yt, yp, gr)
        _, mlow, mhigh = M.cluster_bootstrap_ci(yt[mt], yp[mt], gr[mt])
        m.update({"config": config, "accuracy_ci_low": low, "accuracy_ci_high": high,
                  "matched_accuracy": float(np.mean(yp[mt] == yt[mt])), "matched_n": int(mt.sum()),
                  "matched_ci_low": mlow, "matched_ci_high": mhigh})
        pooled.append(m)
    out = pd.DataFrame(pooled).sort_values("matched_accuracy", ascending=False)
    out.to_csv(OUT_ROOT / "pooled_config_results.csv", index=False)
    print(out[["config", "accuracy", "matched_accuracy", "matched_ci_low", "matched_ci_high",
               "accuracy_ci_low", "accuracy_ci_high"]].to_string(index=False), flush=True)


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
