"""频域 CNN 基线数据集：STFT log 幅度谱 + top-k 句类标签。

本目录自包含，不依赖 ``experiments`` 下其它实验或 ``experiments/deepnet/eegnet``
的代码。数据来源与 EEGNet 目录相同：CSV 选 top-k 类，按 ``(run, pos)`` 从原始
pkl 回取 EEG。

与线性基线的频域特征保持一致的口径：

- Welch/STFT 的窗长都为约 0.512 s：500 Hz 下 ``nperseg=256``，250 Hz 下
  ``nperseg=128``，频率分辨率约 1.95 Hz；
- 只保留 1–45 Hz（若 Nyquist 低于 45 Hz 则截到 Nyquist）；
- 对 log 幅度谱做逐试次、逐通道的 z-score。

返回形状为 ``(C, F, T_frames)``：通道作为 CNN 的输入通道，频率与时间为二维
卷积的两个空间轴。默认 500 Hz，不重采样；``target_sfreq < 500`` 时使用
``resample_poly`` 做带抗混叠的多相重采样。
"""

from __future__ import annotations

import csv
import pickle
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly, stft
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed"
PKL_ROOT = PROJECT_ROOT / "datasets/ds005170-download/derivatives/preprocessed_pkl"

SFREQ = 500.0
N_EEG_CHANNELS = 122
VOLT_TO_MICROVOLT = 1e6
LABEL_COLUMN = "type_number"
RESAMPLE_CHUNK = 50
STFT_CHUNK = 100
FREQ_MIN = 1.0
FREQ_MAX = 45.0
WINDOW_SECONDS = 0.512

TASK_ALIASES = {
    "reading": ("reading", "read"),
    "imagine": ("imagine", "imagine"),
}

_FILE_PATTERN = re.compile(r"_task-(?P<task>[a-z]+)_run-(?P<run>\d+)_eeg\.pkl$")


def read_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resample_eeg(
    data: np.ndarray,
    sfreq_in: float = SFREQ,
    sfreq_out: float = SFREQ,
    chunk: int = RESAMPLE_CHUNK,
) -> np.ndarray:
    """沿最后一维做多相重采样，``resample_poly`` 自带抗混叠 FIR 滤波。"""
    if float(sfreq_out) == float(sfreq_in):
        return data.astype(np.float32, copy=False)

    ratio = Fraction(int(sfreq_out), int(sfreq_in)).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    parts: list[np.ndarray] = []
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        parts.append(
            resample_poly(block, up, down, axis=-1).astype(np.float32, copy=False)
        )
    return np.concatenate(parts, axis=0)


class FreqCNNDataset(Dataset):
    """STFT log 幅度谱 + top-k 句类标签，供频域 CNN 使用。

    Args:
        subject: 被试编号，如 ``sub-01``。
        task: ``imagine`` / ``reading``。
        top_k: 按 ``type_number`` 出现次数保留前 k 类，默认 10。
        target_sfreq: 返回数据的采样率，默认 500 Hz；低于 500 时做多相重采样。
        csv_root: 句表目录。
        pkl_root: ``derivatives/preprocessed_pkl`` 目录。
    """

    def __init__(
        self,
        subject: str = "sub-01",
        task: str = "imagine",
        top_k: int = 10,
        target_sfreq: float = SFREQ,
        csv_root: Path = CSV_ROOT,
        pkl_root: Path = PKL_ROOT,
    ):
        if task.lower() not in TASK_ALIASES:
            raise ValueError(f"task 需为 reading/imagine 之一，收到 {task!r}")
        if not 0 < float(target_sfreq) <= SFREQ:
            raise ValueError(f"target_sfreq 需在 (0, {SFREQ}] 内，收到 {target_sfreq}")
        if not float(target_sfreq).is_integer():
            raise ValueError(f"target_sfreq 需为整数，收到 {target_sfreq}")

        label, pkl_task = TASK_ALIASES[task.lower()]
        self.subject = subject
        self.task = label
        self.pkl_task = pkl_task
        self.top_k = top_k
        self.sfreq = float(target_sfreq)
        self.freq_min = FREQ_MIN
        self.freq_max = min(FREQ_MAX, self.sfreq / 2.0 - 1e-6)
        self.nperseg = max(16, int(round(WINDOW_SECONDS * self.sfreq)))
        self.noverlap = self.nperseg // 2

        # ---------- 1. CSV 选类 ----------
        csv_path = Path(csv_root) / f"{subject}_{label}.csv"
        rows = read_rows(csv_path)
        if not rows:
            raise RuntimeError(f"{csv_path} 没有任何样本")
        missing = [row["index"] for row in rows if not row[LABEL_COLUMN]]
        if missing:
            raise ValueError(f"{csv_path} 有 {len(missing)} 行缺少 {LABEL_COLUMN}")

        counts = Counter(int(row[LABEL_COLUMN]) for row in rows)
        top = counts.most_common(top_k)
        wanted = {class_id for class_id, _ in top}
        class_names = {
            int(row[LABEL_COLUMN]): row.get("type_name", "") for row in rows
        }
        print(f"[CSV] {csv_path} 共 {len(rows)} 条，选出 {len(top)} 类：")
        for rank, (class_id, count) in enumerate(top, start=1):
            print(f"  {rank:>2}. type={class_id:>2}  n={count:>4}  {class_names[class_id]}")

        rows = [row for row in rows if int(row[LABEL_COLUMN]) in wanted]
        rows.sort(
            key=lambda row: (
                int(row[LABEL_COLUMN]),
                int(row["run"]),
                int(row["pos"]),
            )
        )
        self.rows = rows

        self.class_ids = sorted(wanted)
        self.class_names = [class_names[class_id] for class_id in self.class_ids]
        class_to_label = {
            class_id: index for index, class_id in enumerate(self.class_ids)
        }
        self.labels = np.asarray(
            [class_to_label[int(row[LABEL_COLUMN])] for row in rows], dtype=np.int64
        )
        self.targets = torch.from_numpy(self.labels)

        # ---------- 2. run / day 分组 ----------
        raw_runs = [int(row["run"]) for row in rows]
        self.run_ids = sorted(set(raw_runs))
        self.run_groups = np.asarray(raw_runs, dtype=np.int64)
        self.day_groups = np.asarray([int(row["day"]) for row in rows], dtype=np.int64)
        self.meta = [
            {
                "index": int(row["index"]),
                "day": int(row["day"]),
                "run": int(row["run"]),
                "pos": int(row["pos"]),
                "n_chars": int(row["n_chars"]),
            }
            for row in rows
        ]
        self.texts = [row.get("sentence", "") for row in rows]

        # ---------- 3. 按 run 加载原始 EEG ----------
        self.n_channels = N_EEG_CHANNELS
        self.eeg_dir = Path(pkl_root) / subject / "eeg"
        pkl_paths: dict[int, Path] = {}
        for path in self.eeg_dir.glob(f"{subject}_task-{pkl_task}_run-*_eeg.pkl"):
            match = _FILE_PATTERN.search(path.name)
            if match is not None:
                pkl_paths[int(match["run"])] = path

        rows_by_run: dict[int, list[int]] = defaultdict(list)
        for index, run in enumerate(raw_runs):
            rows_by_run[run].append(index)

        data = None
        for run in self.run_ids:
            path = pkl_paths.get(run)
            if path is None:
                raise FileNotFoundError(f"缺少 run-{run} 的 pkl（目录 {self.eeg_dir}）")
            with path.open("rb") as handle:
                items = pickle.load(handle)

            indices = rows_by_run[run]
            for index in indices:
                pos = int(rows[index]["pos"])
                if pos >= len(items):
                    raise IndexError(
                        f"run-{run} 的 pkl 只有 {len(items)} 条，pos={pos} 越界"
                    )
                eeg = np.asarray(items[pos]["input_features"])
                if eeg.ndim == 3 and eeg.shape[0] == 1:
                    eeg = eeg[0]
                if eeg.ndim != 2 or eeg.shape[0] < N_EEG_CHANNELS:
                    raise RuntimeError(
                        f"run-{run} pos={pos} 的 input_features 形状异常: {eeg.shape}"
                    )
                eeg = eeg[:N_EEG_CHANNELS]

                if data is None:
                    data = np.empty(
                        (len(rows), N_EEG_CHANNELS, eeg.shape[-1]), dtype=np.float32
                    )
                elif eeg.shape[-1] != data.shape[-1]:
                    raise RuntimeError(
                        f"run-{run} pos={pos} 的试次长度 {eeg.shape[-1]} "
                        f"与其他试次 {data.shape[-1]} 不一致"
                    )
                data[index] = eeg * VOLT_TO_MICROVOLT  # V -> µV
            print(f"[Load] run-{run:>02} 共 {len(items)} 条，选中 {len(indices)} 条", flush=True)
            del items

        if data is None:
            raise RuntimeError("没有加载到任何试次")

        # ---------- 4. 采样率适配 ----------
        if float(target_sfreq) != SFREQ:
            print(
                f"[Resample] {SFREQ:g} Hz -> {float(target_sfreq):g} Hz "
                f"（resample_poly，带抗混叠）",
                flush=True,
            )
            data = resample_eeg(data, SFREQ, float(target_sfreq))
        self.n_times = int(data.shape[-1])

        # ---------- 5. STFT -> log 幅度谱 -> 逐通道 z-score ----------
        spec_parts: list[np.ndarray] = []
        expected_shape: tuple[int, int, int] | None = None
        for start in range(0, len(data), STFT_CHUNK):
            block = data[start : start + STFT_CHUNK]
            freqs, _, spectrum = stft(
                block,
                fs=self.sfreq,
                window="hann",
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                boundary="zeros",
                padded=True,
                axis=-1,
            )
            mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
            if not np.any(mask):
                raise RuntimeError(
                    f"STFT 后没有落在 {self.freq_min}–{self.freq_max} Hz 的频点"
                )
            # (n, C, F, T_frames)
            spec = np.log10(np.abs(spectrum[:, :, mask, :]) + 1e-12).astype(
                np.float32, copy=False
            )
            # 逐试次、逐通道 z-score；不使用任何跨试次统计量
            mean = spec.mean(axis=(-2, -1), keepdims=True)
            std = spec.std(axis=(-2, -1), keepdims=True)
            spec = (spec - mean) / np.maximum(std, 1e-6)

            if expected_shape is None:
                expected_shape = spec.shape[1:]
            elif spec.shape[1:] != expected_shape:
                raise RuntimeError(
                    f"不同批次的 STFT 形状不一致: {spec.shape[1:]} vs {expected_shape}"
                )
            spec_parts.append(spec.astype(np.float32, copy=False))
            print(
                f"[STFT] {min(start + STFT_CHUNK, len(data))}/{len(data)}",
                flush=True,
            )

        del data
        spectrum_data = np.concatenate(spec_parts, axis=0)
        del spec_parts
        self.data = torch.from_numpy(spectrum_data)
        self.n_freqs = int(spectrum_data.shape[2])
        self.n_frames = int(spectrum_data.shape[3])

        print(
            f"[Dataset] {subject} {label} 共 {len(self.labels)} 条 / "
            f"{len(self.class_ids)} 类 / {len(self.run_ids)} 个 run | "
            f"x {tuple(self.data.shape)} {self.data.dtype} | "
            f"channels={self.n_channels} | sfreq={self.sfreq:g} Hz | "
            f"STFT nperseg={self.nperseg} noverlap={self.noverlap} | "
            f"freq={self.freq_min:g}–{self.freq_max:g} Hz "
            f"({self.n_freqs} bins × {self.n_frames} frames)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        """返回 ``(C, F, T_frames)`` float32 与句类标签。"""
        return self.data[index], int(self.labels[index])


def split_indices(
    labels,
    val_ratio: float = 0.0,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按句类分层的随机划分，默认 8:2（不设 val）。"""
    labels = np.asarray(labels)
    if not 0 <= val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio/test_ratio 需满足 0<=val<1、0<test<1、val+test<1，"
            f"收到 {val_ratio}/{test_ratio}"
        )

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        rng.shuffle(index)
        n = len(index)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 2:
            n_test = max(n_test, 1)
            if val_ratio > 0 and n - n_test >= 2:
                n_val = max(n_val, 1)
        if n_test + n_val >= n:
            n_val = max(n - n_test - 1, 0)
        test_parts.append(index[:n_test])
        val_parts.append(index[n_test : n_test + n_val])
        train_parts.append(index[n_test + n_val :])

    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(val_parts)),
        np.sort(np.concatenate(test_parts)),
    )


def main() -> None:
    """冒烟检查：加载 sub-01，打印形状与一个 batch。"""
    from torch.utils.data import DataLoader

    dataset = FreqCNNDataset("sub-01")
    print(f"类别映射: {dict(zip(range(len(dataset.class_ids)), dataset.class_ids))}")
    print(
        f"n_samples={len(dataset)} n_channels={dataset.n_channels} "
        f"n_freqs={dataset.n_freqs} n_frames={dataset.n_frames} "
        f"sfreq={dataset.sfreq:g}"
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    print(f"batch: x {tuple(x.shape)} {x.dtype} | y {y.tolist()}")


if __name__ == "__main__":
    main()
