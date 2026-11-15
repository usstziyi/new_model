"""数据缓存与按 run 分组的划分工具。

设计原则
--------
1. 原始 ``datasets/top10_imagine/<subject>/class_*.pkl`` 是 500 Hz、单位 V、
   未滤波的 3.3 s 想象期信号。首次使用时统一做
   ``0.5–45 Hz 零相位带通 -> 降采样到 250 Hz -> V 转 µV -> 存 float16``，
   缓存到 ``plan_auto/cache/<subject>/``，后续实验只读缓存（memmap）。
2. 缓存里保留**未做 z-score** 的信号，标准化策略由各实验自己决定，
   避免公共层替实验做不可逆处理。
3. 所有评估划分都按 run 分组：同一个 run 的试次只会出现在训练或测试一侧。
   随机试次划分会因同一 run 内的共模噪声而高估准确率，本目录不使用。

sub-05 的本地 top10 与全局 top10 相差一个类别（缺 14、多 30）。为了让 5 名
被试共用同一套 10 类标签，缓存构建时若缺少 ``class_14.pkl``，会直接回到
``datasets/ds005170-download/derivatives/preprocessed_pkl`` 按 CSV 的
``run + pos`` 现场抽取，结果与 ``scripts/export_top_classes.py`` 一致。
"""

from __future__ import annotations

import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets"
TOP_ROOT = DATASET_ROOT / "top10_imagine"
RAW_PKL_ROOT = DATASET_ROOT / "ds005170-download/derivatives/preprocessed_pkl"
CSV_ROOT = DATASET_ROOT / "preprocessed"
CLASSNUMBER = DATASET_ROOT / "ds005170-download/json/classnumber.json"
CACHE_ROOT = PROJECT_ROOT / "plan_auto/cache"

SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-04", "sub-05")
# 全体被试汇总后样本最多的 10 类（与 sub-01~04 的本地 top10 相同）
CLASS_IDS = (2, 10, 13, 14, 19, 21, 24, 26, 31, 32)

N_CHANNELS = 122
SFREQ_RAW = 500.0
_FILTER = butter(4, (0.5, 45.0), btype="bandpass", fs=SFREQ_RAW, output="sos")
_FILE_PATTERN = re.compile(r"class_(?P<cid>\d+)\.pkl$")
_RAW_PATTERN = re.compile(r"_task-(?P<task>[a-z]+)_run-(?P<run>\d+)_eeg\.pkl$")


# ---------------------------------------------------------------- 基础信息


def load_class_names(path: Path = CLASSNUMBER) -> dict[int, str]:
    """读取 classnumber.json，返回 ``{type_number: 类别名}``。"""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): str(value) for key, value in raw.items()}


def cache_dir(
    subject: str,
    target_sfreq: int = 250,
    root: Path = CACHE_ROOT,
    highpass: float = 0.5,
) -> Path:
    suffix = "" if abs(highpass - 0.5) < 1e-9 else f"_hp{highpass:g}"
    return Path(root) / f"{subject}_sf{target_sfreq}{suffix}"


# ---------------------------------------------------------------- 原始抽取


def _read_csv_rows(subject: str) -> list[dict]:
    path = CSV_ROOT / f"{subject}_imagine.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_from_raw(subject: str, class_id: int) -> dict:
    """从原始 run pkl 抽取某个类别的试次（top10 里缺该类时使用）。"""
    rows = [row for row in _read_csv_rows(subject) if int(row["type_number"]) == class_id]
    if not rows:
        raise FileNotFoundError(f"{subject} 的 CSV 中没有 type_number={class_id}")

    eeg_dir = RAW_PKL_ROOT / subject / "eeg"
    pkl_paths: dict[int, Path] = {}
    for path in eeg_dir.glob(f"{subject}_task-imagine_run-*_eeg.pkl"):
        match = _RAW_PATTERN.search(path.name)
        if match is not None:
            pkl_paths[int(match["run"])] = path

    by_run: dict[int, list[dict]] = {}
    for row in rows:
        by_run.setdefault(int(row["run"]), []).append(row)

    arrays, runs, days, n_chars, texts = [], [], [], [], []
    for run in sorted(by_run):
        with pkl_paths[run].open("rb") as handle:
            items = pickle.load(handle)
        for row in sorted(by_run[run], key=lambda r: int(r["pos"])):
            pos = int(row["pos"])
            features = np.asarray(items[pos]["input_features"], dtype=np.float64)
            if features.ndim == 3 and features.shape[0] == 1:
                features = features[0]
            arrays.append(features[:N_CHANNELS])
            runs.append(run)
            days.append(int(row["day"]))
            n_chars.append(int(row["n_chars"]))
            texts.append(row.get("sentence", ""))

    return {
        "eeg": np.stack(arrays),
        "run": np.asarray(runs, dtype=np.int64),
        "day": np.asarray(days, dtype=np.int64),
        "n_chars": np.asarray(n_chars, dtype=np.int64),
        "text": texts,
        "source": f"raw:{class_id}",
    }


def _load_class(subject: str, class_id: int) -> dict:
    path = TOP_ROOT / subject / f"class_{class_id:02d}.pkl"
    if not path.exists():
        return _extract_from_raw(subject, class_id)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    eeg = np.asarray(payload["eeg"])
    meta = payload.get("meta") or [{} for _ in range(len(eeg))]
    return {
        "eeg": eeg,
        "run": np.asarray([int(m.get("run", -1)) for m in meta], dtype=np.int64),
        "day": np.asarray([int(m.get("day", -1)) for m in meta], dtype=np.int64),
        "n_chars": np.asarray([int(m.get("n_chars", -1)) for m in meta], dtype=np.int64),
        "text": list(payload.get("text", [""] * len(eeg))),
        "source": path.name,
    }


# ---------------------------------------------------------------- 缓存构建


def build_cache(
    subject: str,
    target_sfreq: int = 250,
    root: Path = CACHE_ROOT,
    overwrite: bool = False,
    highpass: float = 0.5,
) -> Path:
    """读取该被试的 10 个类别 pkl，滤波/降采样后写入缓存目录。"""
    if SFREQ_RAW % target_sfreq != 0:
        raise ValueError(f"target_sfreq 必须整除 {SFREQ_RAW}")
    ratio = int(SFREQ_RAW // target_sfreq)
    out_dir = cache_dir(subject, target_sfreq, root, highpass)
    if out_dir.exists() and not overwrite:
        print(f"[Cache] 已存在，跳过：{out_dir}", flush=True)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sos = butter(4, (highpass, 45.0), btype="bandpass", fs=SFREQ_RAW, output="sos")

    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    runs, days, n_chars, texts = [], [], [], []
    lengths: set[int] = set()

    for label, class_id in enumerate(CLASS_IDS):
        data = _load_class(subject, class_id)
        eeg = np.asarray(data["eeg"], dtype=np.float64)
        if eeg.ndim == 4 and eeg.shape[1] == 1:
            eeg = eeg[:, 0]
        if eeg.ndim != 3 or eeg.shape[1] < N_CHANNELS:
            raise RuntimeError(f"{subject} class {class_id} 形状异常: {eeg.shape}")
        eeg = eeg[:, :N_CHANNELS] * 1e6  # V -> µV
        eeg = sosfiltfilt(sos, eeg, axis=-1)
        eeg = np.ascontiguousarray(eeg[:, :, ::ratio], dtype=np.float16)
        lengths.add(eeg.shape[-1])

        arrays.append(eeg)
        labels.append(np.full(len(eeg), label, dtype=np.int64))
        runs.append(data["run"].astype(np.int64))
        days.append(data["day"].astype(np.int64))
        n_chars.append(data["n_chars"].astype(np.int64))
        texts.extend(data["text"])
        print(
            f"[Cache] {subject} class={class_id:<2} ({data['source']}) "
            f"n={len(eeg)} shape={eeg.shape}",
            flush=True,
        )

    if len(lengths) != 1:
        raise RuntimeError(f"{subject} 各类别长度不一致: {sorted(lengths)}")

    eeg = np.concatenate(arrays, axis=0)
    del arrays
    np.save(out_dir / "eeg.npy", eeg)
    np.save(out_dir / "labels.npy", np.concatenate(labels))
    np.save(out_dir / "runs.npy", np.concatenate(runs))
    np.save(out_dir / "days.npy", np.concatenate(days))
    np.save(out_dir / "n_chars.npy", np.concatenate(n_chars))
    (out_dir / "texts.json").write_text(
        json.dumps(texts, ensure_ascii=False), encoding="utf-8"
    )
    names = load_class_names()
    meta = {
        "subject": subject,
        "target_sfreq": target_sfreq,
        "class_ids": list(CLASS_IDS),
        "class_names": [names.get(cid, "") for cid in CLASS_IDS],
        "shape": list(eeg.shape),
        "n_trials": int(eeg.shape[0]),
        "unit": "uV",
        "filter": f"{highpass:g}-45 Hz butterworth(4), sosfiltfilt",
        "normalization": "none (per-experiment)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Cache] {subject} 完成 shape={eeg.shape} "
        f"({out_dir / 'eeg.npy'}) {eeg.nbytes / 1e9:.2f} GB",
        flush=True,
    )
    return out_dir


def load_cache(
    subject: str,
    target_sfreq: int = 250,
    root: Path = CACHE_ROOT,
    highpass: float = 0.5,
) -> dict:
    """memmap 方式读取缓存（不会一次性占满内存）。"""
    data_dir = cache_dir(subject, target_sfreq, root, highpass)
    if not (data_dir / "eeg.npy").exists():
        raise FileNotFoundError(f"缺少缓存 {data_dir}，先运行 build_cache")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "eeg": np.load(data_dir / "eeg.npy", mmap_mode="r"),
        "labels": np.load(data_dir / "labels.npy"),
        "runs": np.load(data_dir / "runs.npy"),
        "days": np.load(data_dir / "days.npy"),
        "n_chars": np.load(data_dir / "n_chars.npy"),
        "texts": json.loads((data_dir / "texts.json").read_text(encoding="utf-8")),
        "meta": meta,
        "path": data_dir,
    }


# ---------------------------------------------------------------- 划分


def grouped_folds(
    run_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 run 分组、尽量平衡试次数的 K 折划分。

    采用 LPT（大 run 先放、每次放到当前最空的折）分配，再对类别覆盖做一次
    修补，保证每个测试折都含全部类别。返回 ``[(train_idx, test_idx), ...]``。
    """
    run_ids = np.asarray(run_ids)
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    unique_runs = np.unique(run_ids)
    if len(unique_runs) < n_splits:
        raise ValueError(f"run 数 {len(unique_runs)} 少于折数 {n_splits}")

    rng = np.random.default_rng(seed)
    shuffled = unique_runs.copy()
    rng.shuffle(shuffled)
    sizes = {int(r): int(np.sum(run_ids == r)) for r in unique_runs}
    order = sorted(shuffled, key=lambda r: -sizes[int(r)])

    loads = np.zeros(n_splits)
    assign: dict[int, int] = {}
    for run in order:
        best = min(range(n_splits), key=lambda f: (loads[f], rng.random()))
        assign[int(run)] = best
        loads[best] += sizes[int(run)]

    # 修补：若某折缺少某类，从别的折搬一个含该类且移动后最不影响均衡的 run
    def fold_classes(fold: int) -> set[int]:
        idx = np.flatnonzero(np.isin(run_ids, [r for r, f in assign.items() if f == fold]))
        return set(np.unique(labels[idx]).tolist())

    for fold in range(n_splits):
        for _ in range(n_classes):
            missing = set(range(n_classes)) - fold_classes(fold)
            if not missing:
                break
            candidates = []
            for run, owner in assign.items():
                if owner == fold:
                    continue
                run_labels = set(np.unique(labels[run_ids == run]).tolist())
                hit = missing & run_labels
                if not hit:
                    continue
                delta = sizes[run]
                candidates.append((loads[fold] + delta - loads[owner] + delta, run, owner))
            if not candidates:
                break
            _, run, owner = min(candidates)
            assign[run] = fold
            loads[fold] += sizes[run]
            loads[owner] -= sizes[run]

    folds = []
    for fold in range(n_splits):
        test_runs = {r for r, f in assign.items() if f == fold}
        test_idx = np.flatnonzero(np.isin(run_ids, list(test_runs)))
        train_idx = np.flatnonzero(~np.isin(run_ids, list(test_runs)))
        folds.append((train_idx, test_idx))
        present = set(np.unique(labels[test_idx]).tolist())
        missing = sorted(set(range(n_classes)) - present)
        if missing:
            print(f"[Folds] 警告：第 {fold} 折缺少类别 {missing}", flush=True)
    return folds


def split_train_val(
    train_idx: np.ndarray,
    run_ids: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """从训练折里再按 run 留出验证集，用于早停/选 epoch。

    先随机抽 run 直到验证集占比接近 ``val_ratio``，再补齐验证集里缺失的类别。
    """
    train_idx = np.asarray(train_idx)
    run_ids = np.asarray(run_ids)
    labels = np.asarray(labels)
    inner_runs = np.unique(run_ids[train_idx])
    rng = np.random.default_rng(seed)
    rng.shuffle(inner_runs)

    n_classes = int(labels.max()) + 1
    target = int(len(train_idx) * val_ratio)
    chosen: list[int] = []
    counts = np.zeros(n_classes)
    for run in inner_runs:
        mask = run_ids[train_idx] == run
        run_counts = np.bincount(labels[train_idx][mask], minlength=n_classes)
        chosen.append(int(run))
        counts += run_counts
        if counts.sum() >= target:
            break

    coverage = {c for c in range(n_classes) if counts[c] > 0}
    for run in inner_runs:
        if len(coverage) == n_classes:
            break
        if int(run) in chosen:
            continue
        run_labels = set(np.unique(labels[train_idx][run_ids[train_idx] == run]).tolist())
        if run_labels - coverage:
            chosen.append(int(run))
            coverage |= run_labels

    val_mask = np.isin(run_ids[train_idx], chosen)
    val_idx = train_idx[val_mask]
    fit_idx = train_idx[~val_mask]
    if len(val_idx) == 0 or len(fit_idx) == 0:
        raise RuntimeError("验证集划分失败")
    return np.sort(fit_idx), np.sort(val_idx)


def length_matched_indices(
    labels: np.ndarray,
    n_chars: np.ndarray,
    seed: int = 42,
    lengths: tuple[int, ...] | None = None,
    max_per_class: int | None = None,
) -> np.ndarray:
    """按句子长度做跨类别配平，得到不受“句长分布差异”影响的评估子集。

    对每个长度值，取各类别在该长度下的最小样本数，等量抽样，使 10 个类别
    的 ``n_chars`` 分布完全一致。这样用句长做捷径不可能高于随机水平。
    """
    labels = np.asarray(labels)
    n_chars = np.asarray(n_chars)
    if lengths is None:
        lengths = tuple(range(int(n_chars.min()), int(n_chars.max()) + 1))
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for length in lengths:
        buckets = []
        for class_id in np.unique(labels):
            idx = np.flatnonzero((labels == class_id) & (n_chars == length))
            buckets.append(idx)
        if not buckets or min(len(b) for b in buckets) == 0:
            continue
        n = min(len(b) for b in buckets)
        for idx in buckets:
            keep.append(rng.choice(idx, size=n, replace=False))
    if not keep:
        raise RuntimeError("没有可用的句长配平子集")
    out = np.sort(np.concatenate(keep))
    if max_per_class is not None:
        # 每类最多保留 max_per_class 条
        picked = []
        for class_id in np.unique(labels[out]):
            idx = out[labels[out] == class_id]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            picked.append(idx)
        out = np.sort(np.concatenate(picked))
    return out
