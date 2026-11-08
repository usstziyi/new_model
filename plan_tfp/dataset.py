"""plan_tfp 数据加载：通道规整 + 试次级数据集。

两条入口：
- ``TFPEEGDataset``（实际在用）：读取 ``datasets/top<k>_imagine/<subject>/
  class_*.pkl``（已由 ``scripts/export_top_classes.py`` 导出为 122 导），按每个
  pkl 内的试次数从多到少取前 7 个类别，并把类别重映射为 ``0..6``。
- ``prepare_raw``（备用，当前训练链路未调用）：把 Chisco 原始 EDF/FIF 规整成
  122 导 EEG（删非 EEG 通道、挂电极坐标、校验顺序）。训练用的 pkl 本来就是
  122 导纯数组，不含通道名与坐标，所以这条路径暂时用不上；等需要从 EDF 重新
  生成数据时再接进来。

通道顺序的唯一真值来自 ``channels_122.csv``：它与 Chisco pkl 张量的 axis-1、
以及 fif 的前 122 个通道逐行一致（fif 的 122..124 是 VEO/HEO/Trigger）。
"""

from __future__ import annotations

import csv
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TOP_ROOT = PROJECT_ROOT / "datasets/top10_imagine"
CHANNELS_CSV = HERE / "channels_122.csv"
MONTAGE_CSV = HERE / "montage.csv"

SFREQ = 500
N_EEG_CHANNELS = 122
N_TIMES = 1651
N_CLASSES = 7          # 按试次数从多到少只取前 7 个类别
VOLT_TO_MICROVOLT = 1e6

# 需要剔除的非 EEG 通道：6 个数字占位电极 + 4 个外部电极 + 刺激标记
NON_EEG_CHANNELS = {
    "10", "11", "84", "85", "110", "111",
    "VEO", "HEO", "EKG", "EMG",
    "Trigger",
}

_FILE_PATTERN = re.compile(r"class_(?P<type_number>\d+)\.pkl$")


@dataclass(frozen=True)
class ChannelLayout:
    """122 个 EEG 通道的名字与坐标，两者同序。"""

    names: tuple[str, ...]
    positions: torch.Tensor          # (C, 3)，单位米


def load_channel_layout(
    channels_csv: Union[str, Path] = CHANNELS_CSV,
    montage_csv: Union[str, Path] = MONTAGE_CSV,
) -> ChannelLayout:
    """读取 pkl 的真实通道顺序，再按名字挂上电极坐标。"""
    with Path(channels_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"index", "name"}.issubset(rows[0]):
        raise ValueError(f"{channels_csv} 表头必须是 index,name")

    names = tuple(row["name"].strip() for row in rows)
    indices = [int(row["index"]) for row in rows]
    if indices != list(range(len(names))):
        raise ValueError("channels csv 的 index 必须是按行递增的 0..N-1")
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("channels csv 存在重复通道名")

    coords: dict[str, tuple[float, float, float]] = {}
    with Path(montage_csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            coords[row["label"].strip().casefold()] = tuple(
                float(row[k]) for k in ("x", "y", "z")
            )

    missing = [n for n in names
               if n.casefold() not in coords
               or math.dist(coords[n.casefold()], (0.0, 0.0, 0.0)) <= 1e-8]
    if missing:
        raise ValueError(f"以下通道在 montage.csv 中没有有效坐标: {missing}")

    pos = torch.tensor([coords[n.casefold()] for n in names], dtype=torch.float32)
    radius = pos.norm(dim=-1)
    if bool((radius < 0.03).any()) or bool((radius > 0.4).any()):
        raise ValueError("montage 坐标必须是米（距原点 0.03-0.4 m）")
    if bool((torch.pdist(pos) < 1e-6).any()):
        raise ValueError("存在坐标完全重合的通道")
    return ChannelLayout(names=names, positions=pos)


def build_adjacency(
    positions: torch.Tensor,
    *,
    sigma_mode: str = "nn",
    nn_factor: float = 2.0,
) -> torch.Tensor:
    """电极距离高斯核 + 行归一化，得到 (C, C) 稠密邻接，供模型做拓扑先验。

    σ 的取法（论文未规定，属重建选择）：

    - ``"nn"``：σ = ``nn_factor`` × 各电极最近邻距离的中位数。默认，邻域结构清晰
      （2× 最近邻下每个电极约 36 个有效邻居）。
    - ``"median"``：σ = 所有电极对距离的中位数。核过宽，122 导时会退化成近似
      均匀矩阵（有效邻居约 111 个），仅作消融基线保留。
    """
    if sigma_mode not in ("nn", "median"):
        raise ValueError(f"sigma_mode 只能是 'nn' 或 'median'，收到 {sigma_mode!r}")
    if nn_factor <= 0:
        raise ValueError(f"nn_factor 必须为正，收到 {nn_factor}")

    dist = torch.cdist(positions, positions)
    if sigma_mode == "nn":
        inf = torch.full_like(dist, float("inf"))
        nearest = torch.where(torch.eye(len(positions), dtype=torch.bool), inf, dist).min(dim=1).values
        sigma = nn_factor * nearest.median()
    else:
        sigma = dist[dist > 0].median()
    sigma = sigma.clamp_min(1e-6)

    adj = torch.exp(-dist.square() / (2.0 * sigma.square()))
    adj.fill_diagonal_(1.0)
    return adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-12)


def expected_channels() -> list[str]:
    """``channels_122.csv`` 给出的 122 个通道名，即 pkl 的 axis-1 顺序。"""
    return list(load_channel_layout().names)


def prepare_raw(raw):
    """删非 EEG 通道、挂电极坐标，并校验 122 导顺序（原地修改并返回）。

    备用入口：当前训练链路读的是已导出的 122 导 pkl，不会调用本函数。等需要从
    原始 EDF/FIF 重新生成数据时再接入。
    """
    import mne

    raw.drop_channels([ch for ch in raw.ch_names if ch in NON_EEG_CHANNELS])
    if len(raw.ch_names) != N_EEG_CHANNELS:
        raise RuntimeError(
            f"剔除非 EEG 通道后应剩 {N_EEG_CHANNELS} 导，实际 {len(raw.ch_names)}"
        )
    if list(raw.ch_names) != expected_channels():
        raise RuntimeError("通道顺序与 channels_122.csv 不一致，禁止继续")

    # montage.csv 没有 nasion/lpa/rpa，MNE 会退化成 identity 变换并告警；
    # 这里只用米制坐标算电极间距，identity 不影响拓扑，故忽略该条告警。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Fiducial point nasion not found")
        raw.set_montage(
            mne.channels.read_custom_montage(MONTAGE_CSV),
            on_missing="raise",
            verbose=False,
        )
    return raw


class TFPEEGDataset(Dataset):
    """拼接各类别 pkl，标签重映射为 ``0..C-1``。

    目录里按试次数从多到少排序后只取前 ``N_CLASSES`` 个 pkl。

    Args:
        subject: 被试编号，如 ``sub-01``。
        top_k: 顶层目录 ``datasets/top<k>_imagine`` 的 k。
        root: 覆盖默认数据根目录。
        normalize: 是否逐试次逐通道 z-score，默认 ``True``。
        sigma_mode: 电极邻接高斯核的 σ 取法，``"nn"`` 或 ``"median"``。
        nn_factor: ``sigma_mode="nn"`` 时 σ = nn_factor × 最近邻距离中位数。

    构造完可读 ``.layout``（通道名与坐标）与 ``.adjacency``（(C,C) 行归一化邻接）。

    ``__getitem__`` 返回 ``(x, label)``，``x`` 为 ``(1, 122, 1651)`` float32。
    """

    def __init__(
        self,
        subject: str = "sub-01",
        top_k: int = 10,
        root: Path | None = None,
        normalize: bool = True,
        dtype: np.dtype = np.float32,
        sigma_mode: str = "nn",
        nn_factor: float = 2.0,
    ):
        class_dir = Path(root or PROJECT_ROOT / f"datasets/top{top_k}_imagine") / subject
        if not class_dir.is_dir():
            raise FileNotFoundError(f"找不到类别目录: {class_dir}")

        self.sigma_mode = sigma_mode
        self.layout = load_channel_layout()
        self.adjacency = build_adjacency(
            self.layout.positions, sigma_mode=sigma_mode, nn_factor=nn_factor,
        )

        paths = sorted(
            p for p in class_dir.glob("class_*.pkl") if _FILE_PATTERN.search(p.name)
        )
        if not paths:
            raise FileNotFoundError(f"{class_dir} 下没有 class_*.pkl")

        # 试次数只能从 pkl 内容读出，所以先全部读入、按试次数从多到少排序，
        # 再只保留前 N_CLASSES 个类别。
        payloads: list[tuple[str, dict]] = []
        for path in paths:
            with path.open("rb") as handle:
                payloads.append((path.name, pickle.load(handle)))
        payloads.sort(key=lambda item: len(item[1]["eeg"]), reverse=True)
        del payloads[N_CLASSES:]

        arrays: list[np.ndarray] = []
        labels: list[int] = []
        self.subject = subject
        self.class_ids: list[int] = []
        self.class_names: list[str] = []
        self.texts: list[str] = []

        for new_label in range(len(payloads)):
            name, payload = payloads.pop(0)      # pop 即释放，避免与 arrays 同时驻留
            eeg = np.asarray(payload["eeg"], dtype=dtype)
            if eeg.ndim != 3 or eeg.shape[1] != N_EEG_CHANNELS:
                raise RuntimeError(f"{name} 形状异常: {eeg.shape}")
            if eeg.shape[2] != N_TIMES:
                raise RuntimeError(f"{name} 试次长度应为 {N_TIMES}，实际 {eeg.shape[2]}")

            texts = list(payload.get("text", [""] * len(eeg)))
            if len(texts) != len(eeg):
                raise RuntimeError(f"{name} 的 text 数量与 eeg 不一致")

            self.class_ids.append(int(payload["type_number"]))
            self.class_names.append(str(payload.get("type_name", "")))
            arrays.append(np.ascontiguousarray(eeg))
            labels.extend([new_label] * len(eeg))
            self.texts.extend(texts)
            print(
                f"[Load] {name} type_number={payload['type_number']} "
                f"-> label={new_label} shape={eeg.shape}",
                flush=True,
            )
        del payloads

        data = np.concatenate(arrays, axis=0)
        del arrays
        data *= VOLT_TO_MICROVOLT
        if normalize:
            data -= data.mean(axis=-1, keepdims=True)
            data /= np.maximum(data.std(axis=-1, keepdims=True), 1e-6)

        self.n_classes = len(self.class_ids)
        self.data = torch.from_numpy(data)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.targets = torch.from_numpy(self.labels)
        print(
            f"[Dataset] {subject} {len(self.labels)} 条 / {self.n_classes} 类 | "
            f"x {tuple(self.data.shape)} | {SFREQ} Hz",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index].unsqueeze(0), int(self.labels[index])


def stratified_folds(labels, n_splits: int = 5, seed: int = 0):
    """被试内分层 k 折，返回 ``[(train_idx, test_idx), ...]``。"""
    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.sort(train), np.sort(test))
        for train, test in splitter.split(np.zeros(len(labels)), labels)
    ]


def main() -> None:
    dataset = TFPEEGDataset("sub-01")
    adj = dataset.adjacency
    eff = 1.0 / (adj ** 2).sum(1)
    print(f"[Layout] {len(dataset.layout.names)} 导 | "
          f"{dataset.layout.names[0]} ... {dataset.layout.names[-1]}")
    print(f"[Adjacency] sigma={dataset.sigma_mode} | {tuple(adj.shape)} | "
          f"有效邻居 {eff.min():.0f}-{eff.max():.0f} (中位 {eff.median():.0f})")
    print(f"类别: {dict(enumerate(zip(dataset.class_ids, dataset.class_names)))}")

    for fold, (train_idx, test_idx) in enumerate(stratified_folds(dataset.labels), 1):
        counts = np.bincount(dataset.labels[test_idx], minlength=dataset.n_classes)
        print(f"fold {fold}: train {len(train_idx)} / test {len(test_idx)} | 每类 {counts.min()}-{counts.max()}")

    x, y = dataset[0]
    print(f"sample: x {tuple(x.shape)} {x.dtype} | label {y}")


if __name__ == "__main__":
    main()
