"""Chisco（OpenNeuro ds005170）句子语义分类 Dataset（单分支模型版）。

每个样本对应 ``datasets/preprocessed/<subject>_<task>.csv`` 里的一行（一句完整中文
句子），以该行的 ``(run, pos)`` 从 ``derivatives/preprocessed_pkl`` 取回 EEG 片段，
再用 ``band.py`` 做零相位带通滤波与首尾裁剪，按 ``bands`` 指定的频段顺序在通道维
拼接，作为单分支模型的输入。标签是该行的 ``type_number``（0~38，共 39 个语义类别）。

数据位置（均相对项目根目录，即本文件所在目录的父目录）::

    datasets/preprocessed/<subject>_<task>.csv
    datasets/ds005170-download/derivatives/preprocessed_pkl/<subject>/eeg/
        <subject>_task-<read|imagine>_run-<NNN>_eeg.pkl
    datasets/ds005170-download/json/classnumber.json

``__getitem__`` 默认返回 ``(x, label)``；``return_sentence=True`` 时额外返回句子文本
``(x, label, sentence)``，供 EEG-文本对比对齐使用：

- ``x``：``torch.float32``，``(len(bands), E, T)``，单位 µV（可选 z-score）。
  默认 ``bands=("alpha", "beta")`` 时为 ``(2, E, T)``；只用一个频段时为
  ``(1, E, T)``。``bands`` 里可包含特殊名 ``"raw"``，表示不做带通滤波的原始
  EEG（仅首尾裁剪，长度与其他频段一致），例如 ``("raw",)`` 或
  ``("raw", "alpha")``。
- ``label``：``torch.long``，句子语义类别 ``type_number``。
- ``sentence``：``str``，CSV 里该试次对应句子的原文。

``E = 122``（``use_64_channels=False``，默认，沿用 pkl 通道顺序）或 ``E = 64``
（``use_64_channels=True``，按 ``channels_64`` 的顺序）。

``T`` 由任务决定、与句长无关：pkl 原始长度 reading 2501 / imagine 1651（500 Hz），
经 ``band.py`` 首尾各裁 0.25 s（共 250 点）后为 2251 / 1401。默认 ``target_sfreq=500``
不做降采样；设为 250 Hz 时长度减半，为 1126 / 701。read 与 imagine 长度不同，
必须分开建 Dataset / 训练，不能混批。

定位方式（CSV 的 ``run`` / ``pos`` 两列）：

- ``run`` -> ``preprocessed_pkl/<subject>/eeg/<subject>_task-<pkl_task>_run-<NNN>_eeg.pkl``
- ``pos`` -> 该 pkl 内 ``list`` 的下标，每项形如
  ``{"text": str, "input_features": (1, 125, T)}``

通道：pkl 的 125 通道 = 122 EEG + ``VEO`` + ``HEO`` + ``Trigger``，顺序沿用 EDF
（已剔除原作者流程里的 10/11/84/85/110/111/EKG/EMG 占位/辅助通道），本 Dataset
只取前 122 个。``Trigger`` 是刺激通道（基线 65280、事件码 65380），与 VEO/HEO
一并剔除，既避免标签泄漏也避免量纲差异。这 122 个通道的 fif 顺序记录在
``scripts/channels_122.csv``（``index,name``），本模块读取它作为通道顺序基准。

``use_64_channels=True`` 时，从这 122 个通道中按 ``channels_64``（前额→枕区的
分层顺序）取出 64 个，并按 ``channels_64`` 的顺序重排后返回；名称匹配不区分
大小写（``channels_64`` 里的 ``FT7`` 对应 fif 里的 ``Ft7``）。

单位：pkl 里的 EEG 采样值沿用 MNE 的默认约定，即**伏特（V）**；这里统一乘
``VOLT_TO_MICROVOLT`` 换算成**微伏（µV）**再返回。

用法（在 ``plan_single_cnn_mlp`` 目录下运行，或把该目录加入 ``sys.path``）::

    from dataset import ChiscoEEGDataset

    ds = ChiscoEEGDataset("sub-01", "imagine", runs=range(1, 28), bands=("alpha",))
"""

from __future__ import annotations

import csv
import json
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from band import BANDS, FS, MARGIN_SEC, ORDER, butter_bandpass, filter_trial

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKL_ROOT = PROJECT_ROOT / "datasets/ds005170-download/derivatives/preprocessed_pkl"
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed"
CLASSNUMBER = PROJECT_ROOT / "datasets/ds005170-download/json/classnumber.json"

SFREQ = FS
# pkl 前 122 个通道是 EEG，后 3 个依次为 VEO / HEO / Trigger
N_EEG_CHANNELS = 122
# pkl 里的采样值是伏特（MNE 约定），统一换算成微伏再返回
VOLT_TO_MICROVOLT = 1e6
# 单分支模型的默认输入频段，在通道维拼接
DEFAULT_BANDS = ("alpha", "beta")
# 特殊频段名：不做带通滤波的原始 EEG（仅首尾裁剪，长度与其他频段一致）
RAW_BAND = "raw"
# CSV 里作为分类标签的列
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

# 122 个 EEG 通道的 fif 顺序（index,name）
CHANNELS_CSV = PROJECT_ROOT / "scripts/channels_122.csv"


def load_channel_names(path: Path = CHANNELS_CSV) -> list[str]:
    """读取 ``channels_122.csv``，返回按 fif 顺序排列的 122 个 EEG 通道名。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        names = [row["name"] for row in csv.DictReader(handle)]
    if len(names) != N_EEG_CHANNELS:
        raise ValueError(
            f"{path} 应有 {N_EEG_CHANNELS} 个通道，实际 {len(names)} 个"
        )
    return names


def resolve_channel_indices(selected: Sequence[str], available: Sequence[str]) -> np.ndarray:
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


def load_class_names(path: Path = CLASSNUMBER) -> dict[int, str]:
    """读取 ``classnumber.json``，返回 ``{type_number: 中文类别名}``。"""
    with Path(path).open(encoding="utf-8") as handle:
        return {int(key): value for key, value in json.load(handle).items()}


def list_runs(subject: str, task: str, csv_root: Path = CSV_ROOT) -> list[int]:
    """返回该 ``(subject, task)`` CSV 中出现的 run 编号（升序去重）。"""
    if task.lower() not in TASK_ALIASES:
        raise ValueError(f"task 需为 reading/imagine 之一，收到 {task!r}")
    label, _ = TASK_ALIASES[task.lower()]
    path = Path(csv_root) / f"{subject}_{label}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sorted({int(row["run"]) for row in csv.DictReader(handle)})


class ChiscoEEGDataset(Dataset):
    """一个 ``(subject, task)`` 组合对应一张 CSV，构成一个 Dataset。

    Args:
        subject: 被试编号，如 ``sub-01``。
        task: ``reading`` / ``imagine``。
        runs: 只保留这些 run（例如训练集 ``range(1, 28)``）；``None`` 表示全部。
        bands: 输入使用的频段及拼接顺序，默认 ``("alpha", "beta")``（2 通道输入）；
            传 ``("alpha",)`` 即严格单频段单通道；也可传特殊名 ``"raw"``，
            表示返回未做带通滤波的原始 EEG（仍按 ``margin_sec`` 首尾裁剪）。
        target_sfreq: 返回数据的采样率，默认 500 Hz（不降采样）；设为 250 Hz
            可把时间长度减半，显著降低显存与训练耗时。
        normalize: 是否对每个试次的每条通道做 z-score（默认 ``True``），
            抵消不同被试 / 不同通道的幅度差异。
        margin_sec: 首尾裁剪时长，默认沿用 ``band.py`` 的 0.25 s。
        use_64_channels: 为 ``True`` 时只取 ``channels_64`` 列出的 64 个通道，
            并按 ``channels_64`` 的顺序返回（``(len(bands), 64, T)``）；默认
            ``False`` 保持原来的 122 通道 pkl/fif 顺序。
        cache_size: 缓存的 run 个数。单个 run 的 pkl 解包后约 300–450 MB，
            配合 ``RunBlockSampler`` 顺序访问时 1 个即可，默认 2 留一点余量。
        csv_root: ``make_text_csv.py`` 输出的 CSV 目录。
        pkl_root: ``derivatives/preprocessed_pkl`` 目录。

    注意：CSV 已由 ``scripts/make_text_csv.py`` 按句子去重（每句只保留首次出现的
    试次），所以一个数据集内的 sentence 唯一；本类不做额外去重。
    """

    def __init__(
        self,
        subject: str,
        task: str,
        runs: Iterable[int] | None = None,
        bands: Sequence[str] = DEFAULT_BANDS,
        target_sfreq: float = 500.0,
        normalize: bool = True,
        margin_sec: float = MARGIN_SEC,
        cache_size: int = 2,
        return_sentence: bool = False,
        csv_root: Path = CSV_ROOT,
        pkl_root: Path = PKL_ROOT,
        use_64_channels: bool = False,
    ):
        if task.lower() not in TASK_ALIASES:
            raise ValueError(f"task 需为 reading/imagine 之一，收到 {task!r}")
        label, pkl_task = TASK_ALIASES[task.lower()]

        bands = tuple(bands)
        if not bands:
            raise ValueError("bands 不能为空")
        unknown = [name for name in bands if name != RAW_BAND and name not in BANDS]
        if unknown:
            raise ValueError(f"未知频段 {unknown}，可选 {[*BANDS, RAW_BAND]}")

        if target_sfreq <= 0 or target_sfreq > SFREQ:
            raise ValueError(f"target_sfreq 需在 (0, {SFREQ}] 内，收到 {target_sfreq}")
        ratio = SFREQ / target_sfreq
        if not ratio.is_integer():
            raise ValueError(f"target_sfreq 需能整除 {SFREQ}，收到 {target_sfreq}")
        self._decimate = int(ratio)

        self.subject = subject
        self.task = label
        self.pkl_task = pkl_task
        self.bands = bands
        self.normalize = normalize
        self.margin_sec = margin_sec
        self.return_sentence = return_sentence

        # 通道顺序基准：channels_122.csv 的 fif 顺序；
        # use_64_channels 时按 channels_64 的顺序取子集
        self.channel_names = load_channel_names()
        if use_64_channels:
            self._channel_indices = resolve_channel_indices(channels_64, self.channel_names)
            self.channel_names = [self.channel_names[i] for i in self._channel_indices]
        else:
            self._channel_indices = None
        self.n_channels = len(self.channel_names)

        csv_path = Path(csv_root) / f"{subject}_{label}.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows: list[dict] = list(csv.DictReader(handle))

        if runs is not None:
            keep = {int(run) for run in runs}
            rows = [row for row in rows if int(row["run"]) in keep]
        if not rows:
            raise RuntimeError(f"{csv_path} 在指定 runs 下没有任何样本")

        missing = [row["index"] for row in rows if not row[LABEL_COLUMN]]
        if missing:
            raise ValueError(
                f"{csv_path} 有 {len(missing)} 行缺少 {LABEL_COLUMN}，"
                "请先重新运行 scripts/make_text_csv.py"
            )

        self.rows = rows
        self.labels = np.asarray([int(row[LABEL_COLUMN]) for row in rows], dtype=np.int64)

        # run -> pkl 路径。文件名是 run-01 / run-010 这种非等宽命名，
        # 直接按名字排序会错位，这里按解析出的 run 编号建索引。
        self.eeg_dir = Path(pkl_root) / subject / "eeg"
        self._pkl_paths: dict[int, Path] = {}
        for path in self.eeg_dir.glob(f"{subject}_task-{pkl_task}_run-*_eeg.pkl"):
            match = _FILE_PATTERN.search(path.name)
            if match is not None:
                self._pkl_paths[int(match["run"])] = path

        self._bands = {name: BANDS[name] for name in bands if name != RAW_BAND}
        self._coeffs = {
            name: butter_bandpass(low, high, SFREQ, ORDER)
            for name, (low, high) in self._bands.items()
        }

        self._cache: OrderedDict[int, list[dict]] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _items(self, run: int) -> list[dict]:
        """取某个 run 的 pkl 内容（带 LRU 缓存）。"""
        if run in self._cache:
            self._cache.move_to_end(run)
            return self._cache[run]

        path = self._pkl_paths.get(run)
        if path is None:
            raise FileNotFoundError(
                f"{self.subject} {self.task} 找不到 run-{run} 的 pkl（目录 {self.eeg_dir}）"
            )
        with path.open("rb") as handle:
            items: list[dict] = pickle.load(handle)

        self._cache[run] = items
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return items

    def _extract_eeg(self, features) -> np.ndarray:
        """``(1, 125, T)`` -> ``(E, T)``：只保留 EEG，并按需取 64 通道子集。"""
        array = np.asarray(features)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[0] < N_EEG_CHANNELS:
            raise RuntimeError(f"无法识别的 input_features 形状: {array.shape}")
        array = array[:N_EEG_CHANNELS]
        if self._channel_indices is not None:
            array = array[self._channel_indices]
        return array

    def _to_tensor(self, band_data: np.ndarray) -> torch.Tensor:
        """V -> µV，可选降采样 / z-score，返回 ``(1, E, T)`` float32。"""
        data = band_data * VOLT_TO_MICROVOLT
        if self._decimate > 1:
            data = data[:, :: self._decimate]
        if self.normalize:
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True)
            data = (data - mean) / np.maximum(std, 1e-6)
        # 滤波与降采样都会产生新数组，这里再保证连续，避免共享 pkl 缓存
        data = np.ascontiguousarray(data, dtype=np.float32)
        return torch.from_numpy(data).unsqueeze(0)

    def __getitem__(self, index: int):
        row = self.rows[index]
        item = self._items(int(row["run"]))[int(row["pos"])]
        eeg = self._extract_eeg(item["input_features"])

        trial = filter_trial(
            eeg,
            fs=SFREQ,
            bands=self._bands,
            order=ORDER,
            margin_sec=self.margin_sec,
            coeffs=self._coeffs,
        )
        # 按 bands 顺序拼成 (len(bands), E, T)
        x = torch.cat([self._to_tensor(trial[name]) for name in self.bands], dim=0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        if self.return_sentence:
            return x, label, row["sentence"]
        return x, label


class RunBlockSampler(Sampler[int]):
    """按 run 分块打乱的采样器：run 顺序随机、run 内下标随机。

    这样既打散了样本顺序，又让属于同一 run 的样本连续读取，使
    ``ChiscoEEGDataset`` 的 pkl LRU 缓存（通常只留 1–2 个 run）持续命中。
    直接用 ``shuffle=True`` 会让每个样本都重载数百 MB 的 pkl。

    Args:
        run_ids: 与 Dataset 下标一一对应的 run 编号列表。
        shuffle: 是否打乱；验证集设为 ``False`` 保持 run 顺序。
        seed: 随机种子；每个 epoch 用 ``seed + epoch`` 派生新顺序。
    """

    def __init__(self, run_ids: Sequence[int], shuffle: bool = True, seed: int = 0):
        self.run_ids = list(run_ids)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        groups: dict[int, list[int]] = {}
        for index, run in enumerate(self.run_ids):
            groups.setdefault(int(run), []).append(index)

        run_keys = list(groups)
        rng = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        if rng is not None:
            run_keys = [run_keys[i] for i in rng.permutation(len(run_keys))]

        indices: list[int] = []
        for run in run_keys:
            block = groups[run]
            if rng is not None:
                block = [block[i] for i in rng.permutation(len(block))]
            indices.extend(block)

        self.epoch += 1
        yield from indices

    def __len__(self) -> int:
        return len(self.run_ids)


def main() -> None:
    """冒烟检查：遍历所有 CSV 建立 dataset，打印规模，并跑一个 batch。"""
    class_names = load_class_names()
    print(f"类别数: {len(class_names)}")

    first = None
    for csv_path in sorted(CSV_ROOT.glob("*.csv")):
        subject, task = csv_path.stem.split("_", 1)
        dataset = ChiscoEEGDataset(subject, task)
        first = first or dataset

        x, target = dataset[0]
        print(
            f"{subject} {task:<8} {len(dataset):>5} 条 | "
            f"x {tuple(x.shape)} bands={dataset.bands} | "
            f"task={int(target)} {class_names.get(int(target), '')}"
        )

    from torch.utils.data import DataLoader

    loader = DataLoader(first, batch_size=4, sampler=RunBlockSampler(
        [int(row["run"]) for row in first.rows], shuffle=True, seed=0
    ))
    x, targets = next(iter(loader))
    print(f"\n{first.subject} {first.task} 取一个 batch (batch_size=4):")
    print(f"  x {tuple(x.shape)} {x.dtype}")
    print(f"  targets {targets.tolist()}")


if __name__ == "__main__":
    main()
