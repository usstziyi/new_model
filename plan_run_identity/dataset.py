"""Top-10 类 EEG 数据集的 run 身份探针版本。

与 ``plan_baseline_lda/dataset.py`` 读取同一批 pkl 和同一套预处理，唯一区别是
**标签不再是句子类别，而是试次所在 run 的编号**（``meta["run"]``）。

用途：检验 EEG 特征里是否存在强烈的 run 特异签名。如果随机划分下 run 能被
轻易解码，说明同一 run 内的试次高度同质，类别分类器有可能借 run 的“指纹”
走捷径。

每个 pkl 的内容::

    {
        "subject": "sub-01", "task": "imagine",
        "type_number": 10, "type_name": "求职和职业发展",
        "sfreq": 500.0, "n_channels": 122, "unit": "V",
        "eeg": np.ndarray,   # (N, 122, T)，单位 V
        "text": list[str],
        "meta": list[dict],  # index / day / run / pos / n_chars
    }

标签构造：把所有类别 pkl 拼起来后，按 ``meta["run"]`` 的升序把原始 run 编号
重映射为连续的 ``0..R-1``；同时保留原始类别标签 ``orig_labels`` 仅用于诊断
run 与句类之间的混淆程度，不参与训练。

用法::

    from dataset import RunLabelEEGDataset, split_indices

    ds = RunLabelEEGDataset("sub-01")
    train_idx, val_idx, test_idx = split_indices(ds.labels, seed=42)
"""

from __future__ import annotations

import csv
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
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


class RunLabelEEGDataset(Dataset):
    """把所有类别 pkl 拼成一个数据集，标签为试次所在 ``meta["run"]``。

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
        raw_runs: list[int] = []
        orig_labels: list[int] = []
        self.orig_class_ids: list[int] = []
        self.orig_class_names: list[str] = []
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

            meta = payload.get("meta", [{} for _ in range(len(eeg))])
            if len(meta) != len(eeg):
                raise RuntimeError(f"{path.name} 的 meta 与 eeg 数量不一致: {len(meta)} vs {len(eeg)}")
            missing_run = [i for i, entry in enumerate(meta) if "run" not in entry]
            if missing_run:
                raise KeyError(f"{path.name} 有 {len(missing_run)} 条 meta 缺少 run 字段")

            self.orig_class_ids.append(int(payload["type_number"]))
            self.orig_class_names.append(str(payload.get("type_name", "")))
            arrays.append(np.ascontiguousarray(eeg))
            raw_runs.extend(int(entry["run"]) for entry in meta)
            orig_labels.extend([new_label] * len(eeg))
            self.texts.extend(payload.get("text", [""] * len(eeg)))
            self.meta.extend(meta)
            print(
                f"[Load] {path.name} type_number={payload['type_number']} "
                f"-> orig_label={new_label} shape={eeg.shape}",
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

        # ---------- run 标签：meta["run"] -> 0..R-1 ----------
        self.run_ids = sorted(set(raw_runs))
        run_to_label = {run: label for label, run in enumerate(self.run_ids)}
        self.run_names = [f"run_{run:02d}" for run in self.run_ids]
        self.labels = np.asarray([run_to_label[run] for run in raw_runs], dtype=np.int64)
        self.targets = torch.from_numpy(self.labels)

        # 原始句类标签，仅用于诊断 run 和句类的混淆
        self.orig_labels = np.asarray(orig_labels, dtype=np.int64)

        print(
            f"[Dataset] {subject} 共 {len(self.labels)} 条 / {len(self.run_ids)} 个 run "
            f"（来自 {len(self.orig_class_names)} 个句类）| "
            f"x {tuple(self.data.shape)} {self.data.dtype} | "
            f"channels={self.n_channels} | sfreq={SFREQ // self._decimate:g} Hz",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        """返回 ``(1, E, T)`` float32 与 run 标签。"""
        return self.data[index].unsqueeze(0), int(self.labels[index])


def split_indices(
    labels,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 run 分层的随机划分，默认 8:1:1。

    逐类（每个 run）独立分配 val/test，保证试次极少的 run 也能分到评估样本；
    这比直接调用 ``train_test_split(stratify=...)`` 更稳，避免最少类只有 1 条时
    二次分层报错。
    """
    labels = np.asarray(labels)
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio/test_ratio 需为正且之和小于 1，收到 {val_ratio}/{test_ratio}"
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
            if n - n_test >= 2:
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


def _display_width(text: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def main() -> None:
    """冒烟检查：加载全量数据，打印 run 分布、划分比例与一个 batch。"""
    from torch.utils.data import DataLoader, Subset

    dataset = RunLabelEEGDataset("sub-01")
    train_idx, val_idx, test_idx = split_indices(dataset.labels)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    counts = {
        split_name: np.bincount(dataset.labels[idx], minlength=len(dataset.run_ids))
        for split_name, idx in splits.items()
    }

    rows = []
    for label, (run, run_name) in enumerate(zip(dataset.run_ids, dataset.run_names)):
        day = dataset.meta[int(np.flatnonzero(dataset.labels == label)[0])].get("day")
        rows.append(f"day{day} {run_name}")

    name_width = max(_display_width(row) for row in rows + ["run"])
    col_width = 8

    def format_row(label: str, values) -> str:
        pad = " " * (name_width - _display_width(label))
        return f"{label}{pad}" + "".join(f"{value:>{col_width}}" for value in values)

    rule = "-" * (name_width + col_width * len(splits))
    print(format_row(label="run", values=list(splits)))
    print(rule)
    for index, row in enumerate(rows):
        print(format_row(label=row, values=[counts[split_name][index] for split_name in splits]))
    print(rule)
    print(format_row(label="汇总", values=[len(idx) for idx in splits.values()]))

    loader = DataLoader(Subset(dataset, train_idx), batch_size=4, shuffle=True)
    x, y = next(iter(loader))
    print(f"batch: x {tuple(x.shape)} {x.dtype} | y(run) {y.tolist()}")


if __name__ == "__main__":
    main()
