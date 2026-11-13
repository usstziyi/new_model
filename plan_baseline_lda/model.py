"""线性基线的特征前端与分类器。

- ``extract_features``：逐通道 Welch 频带功率（相对 / 绝对），得到
  ``(N, 通道数 × 频段数)`` 的 log 功率特征；
- ``make_model``：在标准化特征上拟合收缩 LDA（主基线）或逻辑回归（参考）。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
NPERSEG = 256  # 0.512 s @500 Hz，与前期探针一致


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
        # (n, 122, T)
        block = data[start : start + chunk]
        # psd:(n, 122, 129) ，单位 µV²/Hz
        freqs, psd = welch(
            x=block, 
            fs=sfreq, 
            nperseg=NPERSEG, 
            return_oneided=True,
            axis=-1
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
            ) # (n, 通道数 × 频段数)=(100, 122 * 5)=(100, 610)
        ) # (N, 610)
    return np.concatenate(feats, axis=0)


def make_model(name: str = "lda"):
    """``lda``: 收缩 LDA（主基线）；``logreg``: 逻辑回归（参考）。"""
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
