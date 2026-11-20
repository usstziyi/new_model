"""run-wise 归一化（归纳式）的特征前端与分类器。

特征与分类器与 ``experiments/diagnostics/baseline_lda`` / ``experiments/diagnostics/loro_lodo`` 逐行一致（log 相对频带
功率 + 收缩 LDA），唯一的差别是送入分类器前多一步按 run 的标准化，并且
**μ/σ 的估计严格不使用任何留出试次**：

- 随机 5 折 CV：每折内，每个 run 只用落在训练折的试次估计 μ/σ，再变换该折的
  全部试次；训练折里没有试次的 run 退回该折训练集的整体统计量。
- LORO / LODO：训练组的每个 run 用自身试次估计 μ/σ；被留出的 run 在训练集中
  从未出现，只能用训练集汇总统计量变换，因此**它自身的 run 级偏移不会被去除**。

两步之后都会再用训练集的汇总 μ/σ 平移缩放一次，保证训练与测试同尺度。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
NPERSEG = 256  # 0.512 s @500 Hz，与前期探针一致
EPS = 1e-6


def extract_features(
    data: np.ndarray,
    sfreq: float,
    mode: str = "rel",
    chunk: int = 100,
) -> np.ndarray:
    """逐通道频带功率特征，返回 ``(N, 通道数 × 频段数)``。

    Args:
        data: ``(N, C, T)`` 的 µV 数据。
        sfreq: 采样率。
        mode: ``rel`` 相对功率（除以该通道总功率）或 ``abs`` 绝对功率。
        chunk: 每次进入 Welch 的试次数，控制内存峰值。
    """
    feats: list[np.ndarray] = []
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        freqs, psd = welch(
            x=block,
            fs=sfreq,
            nperseg=NPERSEG,
            return_onesided=True,
            axis=-1,
        )
        psd = psd.astype(np.float32)
        if mode == "rel":
            psd = psd / (psd.sum(axis=-1, keepdims=True) + 1e-30)
        feats.append(
            np.concatenate(
                [
                    np.log10(psd[:, :, (freqs >= lo) & (freqs < hi)].mean(axis=-1) + 1e-30)
                    for lo, hi in BANDS.values()
                ],
                axis=1,
            )  # (n, 通道数 × 频段数) = (100, 610)
        )
    return np.concatenate(feats, axis=0)


def make_model(name: str = "lda"):
    """``lda``: 收缩 LDA（主基线）；``logreg``: 逻辑回归；``svm``: RBF SVM。"""
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
    raise ValueError(f"未知模型: {name}")


def runwise_stats(feats: np.ndarray, groups: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """按 run 汇总 ``(μ, σ)``；试次数少于 2 的 run 不返回，交给调用方兜底。"""
    stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for run in np.unique(groups):
        block = feats[groups == run]
        if len(block) < 2:
            continue
        stats[int(run)] = (block.mean(axis=0), block.std(axis=0))
    return stats


def apply_stats(
    feats: np.ndarray,
    groups: np.ndarray,
    stats: dict[int, tuple[np.ndarray, np.ndarray]],
    fallback_mu: np.ndarray,
    fallback_sd: np.ndarray,
    center_only: bool = False,
) -> np.ndarray:
    """用 ``stats`` 逐 run 标准化；缺少统计量的 run 使用 ``fallback``。

    ``center_only=True`` 时只减均值、不除标准差。
    """
    out = feats.copy()
    for run in np.unique(groups):
        mask = groups == run
        mu, sd = stats.get(int(run), (fallback_mu, fallback_sd))
        denom = 1.0 if center_only else np.maximum(sd, EPS)
        out[mask] = (feats[mask] - mu) / denom
    return out


def normalize_inductive(
    feats: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    center_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """归纳式 run-wise 归一化：μ/σ 只由 ``train_idx`` 估计。

    Args:
        feats: ``(N, D)`` 特征矩阵。
        groups: ``(N,)`` 每个试次所属的 run 编号（逐 run 归一化的分组）。
        train_idx: 训练试次下标，唯一允许进入 μ/σ 估计的样本。
        test_idx: 测试试次下标，只做变换。
        center_only: 只减均值、不除标准差，用于消融 σ 的贡献。

    Returns:
        ``(x_train, x_test)``：逐 run 标准化后再用训练集汇总统计量统一尺度。
    """
    train_groups = groups[train_idx]
    stats = runwise_stats(feats[train_idx], train_groups)
    global_mu = feats[train_idx].mean(axis=0)
    global_sd = feats[train_idx].std(axis=0)

    x_train = apply_stats(
        feats[train_idx], train_groups, stats, global_mu, global_sd, center_only
    )
    x_test = apply_stats(
        feats[test_idx], groups[test_idx], stats, global_mu, global_sd, center_only
    )

    # 训练集汇总统计量：把逐 run 标准化后的训练数据再拉回统一尺度
    pool_mu = x_train.mean(axis=0)
    if center_only:
        x_train = x_train - pool_mu
        x_test = x_test - pool_mu
    else:
        pool_sd = x_train.std(axis=0)
        x_train = (x_train - pool_mu) / np.maximum(pool_sd, EPS)
        x_test = (x_test - pool_mu) / np.maximum(pool_sd, EPS)
    return x_train, x_test
