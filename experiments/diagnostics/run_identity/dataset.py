"""run 身份探针数据集：直接由试次表 CSV + 原始 run pkl 构造。

数据来源（均相对项目根目录）::

    datasets/preprocessed/<subject>_<task>.csv
        每行一句中文句子：index / day / run / pos / sentence / n_chars / type_number / type_name
    datasets/ds005170-download/derivatives/preprocessed_pkl/<subject>/eeg/
        <subject>_task-<read|imagine>_run-<NNN>_eeg.pkl
        每个 pkl 是该 run 的试次列表，每项 {"text": str, "input_features": (1, 125, T)}

不再依赖 ``datasets/top10_imagine/class_*.pkl`` 这一中间产物，流程改为：

1. 读取 CSV 全部行，按 ``type_number`` 出现次数取前 ``top_k`` 类；
2. 只保留这些类的行，按 ``(type_number, run, pos)`` 排序
   —— 与原先“按类别拼接、类内按 run/pos”的顺序完全一致，结果可直接对比；
3. 按 run 分组，每个 run 的 pkl 只解包一次，用 ``(run, pos)`` 定位试次，
   取前 122 个 EEG 通道，拼成 ``(N, 122, T)`` 并常驻内存。

标签构造：把所有选中试次拼起来后，按 ``meta["run"]`` 的升序把原始 run 编号
重映射为连续的 ``0..R-1``；同时保留原始类别标签 ``orig_labels`` 仅用于诊断
run 与句类之间的混淆程度，不参与训练。

通道：pkl 的 125 通道 = 122 EEG + ``VEO`` + ``HEO`` + ``Trigger``，本模块只取
前 122 个，顺序以 ``scripts/channels_122.csv`` 为基准；``use_64_channels=True``
时按 ``channels_64`` 的顺序取 64 个。pkl 采样值为伏特，加载时统一换算成微伏。

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed"
PKL_ROOT = PROJECT_ROOT / "datasets/ds005170-download/derivatives/preprocessed_pkl"
CHANNELS_CSV = PROJECT_ROOT / "scripts/channels_122.csv"

SFREQ = 500
N_EEG_CHANNELS = 122
# pkl 里的采样值是伏特（MNE 约定），网络侧统一用微伏
VOLT_TO_MICROVOLT = 1e6
# CSV 里作为句类标签的列
LABEL_COLUMN = "type_number"

# 传入的 task -> (CSV 文件名后缀, pkl 文件名里的 task 字段)
TASK_ALIASES = {
    "reading": ("reading", "read"),
    "imagine": ("imagine", "imagine"),
}

_FILE_PATTERN = re.compile(r"_task-(?P<task>[a-z]+)_run-(?P<run>\d+)_eeg\.pkl$")

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
    if len(set(indices)) != len(indices):
        raise ValueError("channels_64 中存在重复通道")
    return np.asarray(indices, dtype=np.int64)


def read_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class RunLabelEEGDataset(Dataset):
    """由 CSV 选类并加载 EEG，标签为试次所在 ``meta["run"]``。

    Args:
        subject: 被试编号，如 ``sub-01``。
        task: ``imagine`` / ``reading``，决定读哪张 CSV 与哪组 run pkl。
        top_k: 按 ``type_number`` 出现次数保留前 k 类，默认 10。
        target_sfreq: 返回数据的采样率，默认 500 Hz（不降采样）。
        normalize: 是否对每个试次的每条通道做 z-score，默认 ``True``。
        use_64_channels: 为 ``True`` 时只取 ``channels_64`` 的 64 个通道。
        dtype: 内存中保存的数据类型，默认 ``float32``。
        csv_root: 试次表目录。
        pkl_root: ``derivatives/preprocessed_pkl`` 目录。
    """

    def __init__(
        self,
        subject: str = "sub-01",
        task: str = "imagine",
        top_k: int = 10,
        target_sfreq: float = 500.0,
        normalize: bool = True,
        use_64_channels: bool = False,
        dtype: np.dtype = np.float32,
        csv_root: Path = CSV_ROOT,
        pkl_root: Path = PKL_ROOT,
    ):
        if task.lower() not in TASK_ALIASES:
            raise ValueError(f"task 需为 reading/imagine 之一，收到 {task!r}")
        label, pkl_task = TASK_ALIASES[task.lower()]

        if target_sfreq <= 0 or target_sfreq > SFREQ:
            raise ValueError(f"target_sfreq 需在 (0, {SFREQ}] 内，收到 {target_sfreq}")
        ratio = SFREQ / target_sfreq
        if not ratio.is_integer():
            raise ValueError(f"target_sfreq 需能整除 {SFREQ}，收到 {target_sfreq}")
        self._decimate = int(ratio)

        self.subject = subject
        self.task = label
        self.pkl_task = pkl_task
        self.top_k = top_k
        self.normalize = normalize
        self._dtype = np.dtype(dtype)

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
        name_width = max(_display_width(class_names[class_id]) for class_id, _ in top)
        header = f"{'rank':>4}  {'class':<{name_width}}  {'type':>4}  {'count':>5}"
        print(header)
        print("-" * len(header))
        for rank, (class_id, count) in enumerate(top, start=1):
            name = class_names[class_id]
            pad = " " * (name_width - _display_width(name))
            print(f"{rank:>4}  {name}{pad}  {class_id:>4}  {count:>5}")

        rows = [row for row in rows if int(row[LABEL_COLUMN]) in wanted]
        # 与原先 top10 pkl（按类别拼接、类内 run/pos 升序）保持同一顺序
        rows.sort(
            key=lambda row: (
                int(row[LABEL_COLUMN]),
                int(row["run"]),
                int(row["pos"]),
            )
        )
        self.rows = rows

        # 原始句类标签 -> 连续编号（按 type_number 升序，便于阅读与对比）
        self.orig_class_ids = sorted(wanted)
        self.orig_class_names = [class_names[class_id] for class_id in self.orig_class_ids]
        class_to_label = {
            class_id: index for index, class_id in enumerate(self.orig_class_ids)
        }

        # ---------- 2. run 标签 ----------
        raw_runs = [int(row["run"]) for row in rows]
        self.run_ids = sorted(set(raw_runs))
        run_to_label = {run: index for index, run in enumerate(self.run_ids)}
        self.run_names = [f"run_{run:02d}" for run in self.run_ids]
        self.labels = np.asarray([run_to_label[run] for run in raw_runs], dtype=np.int64)
        self.targets = torch.from_numpy(self.labels)
        self.orig_labels = np.asarray(
            [class_to_label[int(row[LABEL_COLUMN])] for row in rows], dtype=np.int64
        )

        # 仅用于诊断，不参与训练
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

        # ---------- 3. 通道基准 ----------
        self.channel_names = load_channel_names()
        if use_64_channels:
            self._channel_indices = resolve_channel_indices(channels_64, self.channel_names)
            self.channel_names = [self.channel_names[i] for i in self._channel_indices]
        else:
            self._channel_indices = None
        self.n_channels = len(self.channel_names)

        # ---------- 4. 按 run 加载 EEG ----------
        self.eeg_dir = Path(pkl_root) / subject / "eeg"
        pkl_paths: dict[int, Path] = {}
        for path in self.eeg_dir.glob(f"{subject}_task-{pkl_task}_run-*_eeg.pkl"):
            match = _FILE_PATTERN.search(path.name)
            if match is not None:
                pkl_paths[int(match["run"])] = path

        # run 编号 -> 样本在 rows/data 中的下标；按 run 解包 pkl 时只取需要的试次
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
                if self._channel_indices is not None:
                    eeg = eeg[self._channel_indices]
                if self._decimate > 1:
                    eeg = eeg[:, :: self._decimate]

                if data is None:
                    data = np.empty(
                        (len(rows), self.n_channels, eeg.shape[-1]), dtype=self._dtype
                    )
                elif eeg.shape[-1] != data.shape[-1]:
                    raise RuntimeError(
                        f"run-{run} pos={pos} 的试次长度 {eeg.shape[-1]} "
                        f"与其他试次 {data.shape[-1]} 不一致"
                    )
                data[index] = eeg * VOLT_TO_MICROVOLT  # V -> µV，写入时转 dtype
            print(f"[Load] run-{run:>02} 共 {len(items)} 条，选中 {len(indices)} 条", flush=True)
            del items

        if data is None:
            raise RuntimeError("没有加载到任何试次")

        if self.normalize:
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True)
            data -= mean
            data /= np.maximum(std, 1e-6)

        self.data = torch.from_numpy(data)

        print(
            f"[Dataset] {subject} {label} 共 {len(self.labels)} 条 / {len(self.run_ids)} 个 run "
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
    val_ratio: float = 0.0,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 run 分层的随机划分，默认 8:2（train:test = 4:1，不设 val）。

    逐类（每个 run）独立分配 test（以及可选的 val），保证试次极少的 run 也能
    分到评估样本；这比直接调用 ``train_test_split(stratify=...)`` 更稳，避免
    最少类只有 1 条时二次分层报错。``val_ratio=0`` 时 val 返回空数组。
    """
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


def _display_width(text: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def main() -> None:
    """冒烟检查：加载全量数据，打印 run 分布、划分比例与一个 batch。"""
    from torch.utils.data import DataLoader, Subset

    dataset = RunLabelEEGDataset("sub-02")
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
