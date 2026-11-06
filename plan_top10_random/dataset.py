"""Top-10 类 EEG 数据集（按类别导出的 pkl，随机划分实验）。

读取 ``datasets/top10_imagine/<subject>/class_<id>.pkl``（由
``scripts/export_top_classes.py`` 生成），把所有类别拼成一个数据集，并把原始
``type_number`` 重映射为连续的 ``0..C-1`` 标签。

每个 pkl 的内容::

    {
        "subject": "sub-01", "task": "imagine",
        "type_number": 10, "type_name": "求职和职业发展",
        "sfreq": 500.0, "n_channels": 122, "unit": "V",
        "eeg": np.ndarray,   # (N, 122, T)，单位 V
        "text": list[str],
        "meta": list[dict],  # index / day / run / pos / n_chars
    }

与 ``plan_classification`` 的区别：不再按 day / run 取 pkl，而是一次性把所有
试次读进内存，随机分层划分为 train / val / test（默认 8:1:1）。``meta`` 里的
day / run 只保留作事后诊断，不参与划分。

内存：2459 × 122 × 1651 × 4B ≈ 2 GB，跑之前确认机器有余量。

用法::

    from dataset import TopClassEEGDataset, split_indices

    ds = TopClassEEGDataset("sub-01")
    train_idx, val_idx, test_idx = split_indices(ds.labels, seed=42)
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOP_ROOT = PROJECT_ROOT / "datasets/top10_imagine"
CHANNELS_CSV = PROJECT_ROOT / "scripts/channels_122.csv"

SFREQ = 500
N_EEG_CHANNELS = 122
# pkl 里的采样值是伏特（MNE 约定），网络侧统一用微伏
VOLT_TO_MICROVOLT = 1e6

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


class TopClassEEGDataset(Dataset):
    """把所有类别 pkl 拼成一个数据集，标签重映射为 ``0..C-1``。

    Args:
        subject: 被试编号，如 ``sub-01``。
        root: ``top10_imagine`` 的根目录。
        target_sfreq: 返回数据的采样率，默认 500 Hz（不降采样）。
        normalize: 是否对每个试次的每条通道做 z-score，默认 ``True``。
        use_64_channels: 为 ``True`` 时只取 ``channels_64`` 的 64 个通道。
        dtype: 内存中保存的数据类型，默认 ``float32``。
    """

    def __init__(
        self,
        subject: str = "sub-01",
        root: Path = TOP_ROOT,
        target_sfreq: float = 500.0,
        normalize: bool = True,
        use_64_channels: bool = False,
        dtype: np.dtype = np.float32,
    ):
        if target_sfreq <= 0 or target_sfreq > SFREQ:
            raise ValueError(f"target_sfreq 需在 (0, {SFREQ}] 内，收到 {target_sfreq}")
        ratio = SFREQ / target_sfreq
        if not ratio.is_integer():
            raise ValueError(f"target_sfreq 需能整除 {SFREQ}，收到 {target_sfreq}")
        self._decimate = int(ratio)

        self.subject = subject
        self.normalize = normalize

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

        arrays: list[np.ndarray] = []
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

            self.class_ids.append(int(payload["type_number"]))
            self.class_names.append(str(payload.get("type_name", "")))
            arrays.append(np.ascontiguousarray(eeg))
            labels.extend([new_label] * len(eeg))
            self.texts.extend(payload.get("text", [""] * len(eeg)))
            self.meta.extend(payload.get("meta", [{} for _ in range(len(eeg))]))
            print(
                f"[Load] {path.name} type_number={payload['type_number']} "
                f"-> label={new_label} shape={eeg.shape}",
                flush=True,
            )

        lengths = {array.shape[-1] for array in arrays}
        if len(lengths) != 1:
            raise RuntimeError(f"各类别试次长度不一致: {sorted(lengths)}")

        data = np.concatenate(arrays, axis=0)
        del arrays
        data *= VOLT_TO_MICROVOLT  # V -> µV，原地修改

        if self.normalize:
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True)
            data -= mean
            data /= np.maximum(std, 1e-6)

        self.data = torch.from_numpy(data)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.targets = torch.from_numpy(self.labels)

        print(
            f"[Dataset] {subject} 共 {len(self.labels)} 条 / {len(self.class_ids)} 类 | "
            f"x {tuple(self.data.shape)} {self.data.dtype} | "
            f"channels={self.n_channels} | sfreq={SFREQ // self._decimate:g} Hz",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        """返回 ``(1, E, T)`` float32 与重映射后的标签。"""
        return self.data[index].unsqueeze(0), int(self.labels[index])


def split_indices(
    labels,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层的随机划分，默认 8:1:1。

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

    dataset = TopClassEEGDataset("sub-01")
    print(f"类别映射: {dict(zip(range(len(dataset.class_ids)), dataset.class_ids))}")
    print(f"类别名称: {dict(enumerate(dataset.class_names))}")

    train_idx, val_idx, test_idx = split_indices(dataset.labels)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = np.bincount(dataset.labels[idx], minlength=len(dataset.class_ids))
        print(f"{name:<5} {len(idx):>5} 条 | 每类 {counts.min()}-{counts.max()}")

    loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print(f"batch: x {tuple(x.shape)} {x.dtype} | y {y.tolist()}")


if __name__ == "__main__":
    main()
