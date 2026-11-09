"""plan_double：时域 + 频域（PSD）双分支的 top-10 类 EEG 数据集。

数据来源与划分方式沿用 ``plan_top10_random``：读取
``datasets/top10_imagine/<subject>/class_<id>.pkl``，把所有类别拼成一个数据集，
原始 ``type_number`` 重映射为连续的 ``0..C-1``；划分用 :func:`split_indices`
（按类别分层随机，默认 8:1:1）。

与单分支版本的区别只在 ``__getitem__`` 的返回值——每个样本给两路输入：

- ``x_time``: ``(C, T)``，陷波 + 0.5–70 Hz 零相位带通 + 逐通道 z-score 后的时域 EEG
- ``x_psd`` : ``(C, F)``，同一信号经 Welch 法得到的 F 个频段功率（默认取 log10）

两路都在 ``__init__`` 里预计算。PSD 只有 2459×122×5×4B ≈ 6 MB，时域信号
2459×122×1651×4B ≈ 2 GB（与单分支版本相同），跑之前确认内存余量。

本目录自带 ``channels_122.csv`` 和 ``psd.py``，不依赖其他 ``plan_*`` 目录。

用法::

    from dataset import DualBranchEEGDataset, split_indices

    ds = DualBranchEEGDataset("sub-01")
    (x_time, x_psd), label = ds[0]
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from psd import BANDS as PSD_BANDS, NPERSEG, eeg_band_psd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOP_ROOT = PROJECT_ROOT / "datasets/top10_imagine"
# 本目录副本，保证实验之间解耦
CHANNELS_CSV = Path(__file__).resolve().parent / "channels_122.csv"

SFREQ = 500.0
N_EEG_CHANNELS = 122
# pkl 里的采样值是伏特（MNE 约定），网络侧统一用微伏
VOLT_TO_MICROVOLT = 1e6
# 时域分支的带通范围（覆盖 PSD 的 0.5–70 Hz 五个频段）
FILTER_BAND = (0.5, 70.0)
FILTER_ORDER = 4
NOTCH_FREQS = (50.0, 100.0)
# filtfilt 分块行数，避免一次性为整类数据开 float64 临时数组
FILTER_CHUNK = 2048
# PSD 分块试次数
PSD_CHUNK = 64

_FILE_PATTERN = re.compile(r"class_(?P<type_number>\d+)\.pkl$")

channels_64 = [
    # 额极层（最前）
    "FP1", "FPz", "Fp2",
    # 前额层
    "AF3", "AFz", "AF4",
    # 额区层
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    # 额中央层
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    # 中央-颞层
    "FTT9h", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    # 中央顶-颞后层
    "TTP7h", "TP7", "CPP5h", "CP3", "CP1", "CPPz", "CP2", "CP4", "CPP6h", "TP8", "TTP8h",
    # 顶区层
    "P7", "P5", "P3", "POz", "P4", "P6", "P8",
    # 顶枕层
    "PPO7", "PO3", "POO3", "Oz", "POO4", "PO4", "PPO8",
    # 枕区层（最后）
    "OI1", "Iz", "OI2",
]


def load_channel_names(path: Path = CHANNELS_CSV) -> list[str]:
    """读取 ``channels_122.csv``，返回按 fif 顺序排列的 122 个 EEG 通道名。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        names = [row["name"] for row in csv.DictReader(handle)]
    if len(names) != N_EEG_CHANNELS:
        raise ValueError(f"{path} 应有 {N_EEG_CHANNELS} 个通道，实际 {len(names)} 个")
    return names


def resolve_channel_indices(selected, available) -> np.ndarray:
    """返回 ``selected`` 中每个通道在 ``available`` 里的下标（名称不区分大小写）。"""
    lookup = {name.lower(): index for index, name in enumerate(available)}
    indices: list[int] = []
    missing: list[str] = []
    for name in selected:
        index = lookup.get(name.lower())
        if index is None:
            missing.append(name)
        else:
            indices.append(index)
    if missing:
        raise ValueError(f"以下通道在 fif 顺序中不存在: {missing}")
    return np.asarray(indices, dtype=np.int64)


def filter_eeg(eeg, fs: float = SFREQ, band=FILTER_BAND, order: int = FILTER_ORDER,
               notch_freqs=NOTCH_FREQS, chunk: int = FILTER_CHUNK) -> np.ndarray:
    """对 ``(..., T)`` 的 EEG 做陷波 + 零相位带通，返回 ``float32``。

    按 ``chunk`` 行分块 filtfilt，峰值内存只与单块有关。
    """
    data = np.asarray(eeg, dtype=np.float64)
    flat = data.reshape(-1, data.shape[-1])
    out = np.empty(flat.shape, dtype=np.float32)

    notches = [
        iirnotch(freq, 30.0, fs) for freq in notch_freqs if freq < fs / 2
    ]
    low, high = band
    if not 0 < low < high < fs / 2:
        raise ValueError(f"带通范围非法: {low}-{high} Hz (Nyquist {fs / 2} Hz)")
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")

    for start in range(0, flat.shape[0], chunk):
        block = flat[start:start + chunk]
        for b_notch, a_notch in notches:
            block = filtfilt(b_notch, a_notch, block, axis=-1)
        out[start:start + chunk] = filtfilt(b, a, block, axis=-1)
    return out.reshape(data.shape)


class DualBranchEEGDataset(Dataset):
    """时域 + PSD 双输入的 top-10 类数据集。

    Args:
        subject: 被试编号，如 ``sub-01``。
        root: ``top10_imagine`` 的根目录。
        target_sfreq: 返回数据的采样率，默认 500 Hz（不降采样）；250 Hz 可省一半内存。
        normalize: 是否对每个试次的每条通道做 z-score，默认 ``True``。
        use_64_channels: 为 ``True`` 时只取 ``channels_64`` 的 64 个通道。
        log_psd: PSD 是否取 ``log10``，默认 ``True``（压缩 γ 与 δ 的量级差）。
        nperseg: Welch 窗长，默认 500（1 s，分辨率 1 Hz）。
        dtype: 时域数据在内存中的类型，默认 ``float32``。
    """

    def __init__(
        self,
        subject: str = "sub-01",
        root: Path = TOP_ROOT,
        target_sfreq: float = SFREQ,
        normalize: bool = True,
        use_64_channels: bool = False,
        log_psd: bool = True,
        nperseg: int = NPERSEG,
        dtype: np.dtype = np.float32,
    ):
        if target_sfreq <= 0 or target_sfreq > SFREQ:
            raise ValueError(f"target_sfreq 需在 (0, {SFREQ}] 内，收到 {target_sfreq}")
        ratio = SFREQ / target_sfreq
        if not ratio.is_integer():
            raise ValueError(f"target_sfreq 需能整除 {SFREQ}，收到 {target_sfreq}")
        self._decimate = int(ratio)
        self.sfreq = SFREQ / self._decimate

        self.subject = subject
        self.normalize = normalize
        self.log_psd = log_psd
        self.nperseg = nperseg

        self.class_dir = Path(root) / subject
        if not self.class_dir.is_dir():
            raise FileNotFoundError(f"找不到类别目录: {self.class_dir}")
        paths = sorted(
            (path for path in self.class_dir.glob("class_*.pkl") if _FILE_PATTERN.search(path.name)),
            key=lambda path: int(_FILE_PATTERN.search(path.name)["type_number"]),
        )
        if not paths:
            raise FileNotFoundError(f"{self.class_dir} 下没有 class_*.pkl")

        # 通道顺序基准：channels_122.csv；use_64_channels 时取子集并重排
        self.channel_names = load_channel_names()
        if use_64_channels:
            self._channel_indices = resolve_channel_indices(channels_64, self.channel_names)
            self.channel_names = [self.channel_names[i] for i in self._channel_indices]
        else:
            self._channel_indices = None
        self.n_channels = len(self.channel_names)
        self.num_freq_bands = len(PSD_BANDS)

        time_arrays: list[np.ndarray] = []
        psd_arrays: list[np.ndarray] = []
        labels: list[int] = []
        self.class_ids: list[int] = []
        self.class_names: list[str] = []
        self.texts: list[str] = []
        self.meta: list[dict] = []

        for new_label, path in enumerate(paths):
            with path.open("rb") as handle:
                payload = pickle.load(handle)

            eeg = np.asarray(payload["eeg"], dtype=dtype)
            if eeg.ndim != 3 or eeg.shape[1] < N_EEG_CHANNELS:
                raise RuntimeError(f"{path.name} 的 eeg 形状异常: {eeg.shape}")
            eeg = eeg[:, :N_EEG_CHANNELS]
            if self._channel_indices is not None:
                eeg = eeg[:, self._channel_indices, :]
            if self._decimate > 1:
                eeg = eeg[:, :, :: self._decimate]
            eeg = eeg * VOLT_TO_MICROVOLT  # V -> µV

            # ---- 时域分支：陷波 + 带通 + 可选 z-score ----
            x_time = filter_eeg(eeg, fs=self.sfreq)
            if self.normalize:
                mean = x_time.mean(axis=-1, keepdims=True)
                std = x_time.std(axis=-1, keepdims=True)
                x_time = (x_time - mean) / np.maximum(std, 1e-6)

            # ---- 频域分支：同一信号做 Welch PSD（已陷波，故 notch=False）----
            x_psd = np.concatenate(
                [
                    eeg_band_psd(
                        x_time[start:start + PSD_CHUNK],
                        fs=self.sfreq,
                        bands=PSD_BANDS,
                        nperseg=self.nperseg,
                        notch=False,
                        log=self.log_psd,
                    )
                    for start in range(0, len(x_time), PSD_CHUNK)
                ],
                axis=0,
            )

            self.class_ids.append(int(payload["type_number"]))
            self.class_names.append(str(payload.get("type_name", "")))
            time_arrays.append(np.ascontiguousarray(x_time, dtype=dtype))
            psd_arrays.append(np.ascontiguousarray(x_psd, dtype=np.float32))
            labels.extend([new_label] * len(eeg))
            self.texts.extend(payload.get("text", [""] * len(eeg)))
            self.meta.extend(payload.get("meta", [{} for _ in range(len(eeg))]))
            print(
                f"[Load] {path.name} type_number={payload['type_number']} "
                f"-> label={new_label} time={x_time.shape} psd={x_psd.shape}",
                flush=True,
            )

        lengths = {array.shape[-1] for array in time_arrays}
        if len(lengths) != 1:
            raise RuntimeError(f"各类别试次长度不一致: {sorted(lengths)}")

        time_data = np.concatenate(time_arrays, axis=0)
        del time_arrays
        psd_data = np.concatenate(psd_arrays, axis=0)
        del psd_arrays

        if psd_data.shape[1:] != (self.n_channels, self.num_freq_bands):
            raise RuntimeError(f"PSD 形状异常: {psd_data.shape}")

        self.time_data = torch.from_numpy(time_data)
        self.psd_data = torch.from_numpy(psd_data)
        self.time_len = self.time_data.shape[-1]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.targets = torch.from_numpy(self.labels)

        print(
            f"[Dataset] {subject} 共 {len(self.labels)} 条 / {len(self.class_ids)} 类 | "
            f"time {tuple(self.time_data.shape)} | psd {tuple(self.psd_data.shape)} | "
            f"channels={self.n_channels} | sfreq={self.sfreq:g} Hz",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        """返回 ``((C, T), (C, F))`` 与重映射后的标签。"""
        return (self.time_data[index], self.psd_data[index]), int(self.labels[index])


def split_indices(
    labels,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层的随机划分，默认 8:1:1（与 ``plan_top10_random`` 一致）。

    先切出 ``val_ratio + test_ratio`` 的临时集，再在临时集内按 ``val:test``
    比例二次分层，保证每类在两个集合里的比例一致。
    """
    labels = np.asarray(labels)
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio/test_ratio 需为正且之和小于 1，收到 {val_ratio}/{test_ratio}"
        )

    indices = np.arange(len(labels))
    holdout = val_ratio + test_ratio
    train_idx, rest_idx = train_test_split(
        indices, test_size=holdout, stratify=labels, random_state=seed
    )
    val_share = val_ratio / holdout
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=1.0 - val_share,
        stratify=labels[rest_idx],
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def main() -> None:
    """冒烟检查：加载全量数据，打印划分比例与一个 batch。"""
    from torch.utils.data import DataLoader, Subset

    dataset = DualBranchEEGDataset("sub-01")
    print(f"类别映射: {dict(zip(range(len(dataset.class_ids)), dataset.class_ids))}")
    print(f"类别名称: {dict(enumerate(dataset.class_names))}")

    train_idx, val_idx, test_idx = split_indices(dataset.labels)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = np.bincount(dataset.labels[idx], minlength=len(dataset.class_ids))
        print(f"{name:<5} {len(idx):>5} 条 | 每类 {counts.min()}-{counts.max()}")

    loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True)
    (x_time, x_psd), y = next(iter(loader))
    print(f"batch: time {tuple(x_time.shape)} | psd {tuple(x_psd.shape)} | y {y.tolist()}")
    print(f"psd 统计: min={x_psd.min():.3f} max={x_psd.max():.3f} mean={x_psd.mean():.3f}")


if __name__ == "__main__":
    main()
