"""run-wise 归一化（直推式）的特征前端与分类器。

特征与分类器与 ``plan_baseline_lda`` / ``plan_loro_lodo`` 逐行一致（log 相对频带
功率 + 收缩 LDA），唯一的差别是送入分类器前多一步按 run 的标准化。

直推式的含义：**每个 run 用自身全部试次的 μ/σ 归一化**，不做任何保留。因此在
留出协议下，测试 run 的无标签试次参与了 μ/σ 的估计；在随机 5 折 CV 下，测试
折的试次也参与了自己所在 run 的 μ/σ 估计——这是本实验要量化的口径，必须与
``plan_runwise_inductive`` 并列报告，不能当作与归纳式结果可比的数字。

由于归一化只依赖特征与 run 分组、不依赖标签，三种协议（cv5 / loro / lodo）
共用同一份归一化后的特征矩阵。
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


def normalize_per_run(feats: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """直推式 run-wise 归一化：每个 run 用自身全部试次的 μ/σ。

    Args:
        feats: ``(N, D)`` 特征矩阵。
        groups: ``(N,)`` 每个试次所属的 run 编号。

    注意：调用方一旦把测试试次放进 ``feats``，这些试次就会参与自己所在 run 的
    μ/σ 估计。这正是直推式的定义，不是 bug。
    """
    out = feats.copy()
    for run in np.unique(groups):
        mask = groups == run
        block = feats[mask]
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        out[mask] = (block - mu) / np.maximum(sd, EPS)
    return out
