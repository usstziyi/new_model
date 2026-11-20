"""正交化（直推式）实验的特征前端、run 方向估计与投影。

特征与分类器与 ``experiments/diagnostics/baseline_lda`` / ``experiments/diagnostics/loro_lodo`` 保持一致：逐通道
5 个频段的 Welch log 相对功率 + 收缩 LDA。本模块额外提供三步：

1. ``fit_run_lda``：在给定特征上拟合 run 身份分类器（多类 LDA）；
2. ``directions_from_lda``：取判别力最强的 ``k`` 个方向并正交化；
3. ``project``：把特征投影到这些方向的正交补空间。

直推式的含义：方向用**全部试次**拟合，包括留出 run 的已知分组与无标签特征，
再投影全部试次。这里不使用任何句类标签，但测试 run 的特征参与了方向估计；
因此 cv5 下这一口径同时包含随机划分与测试试次参与投影的双重泄漏，只能作为
诊断列，不能与 LORO 直推式混为一谈。
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


def fit_run_lda(feats: np.ndarray, run_groups: np.ndarray):
    """在给定特征上拟合 run 身份 LDA，返回 ``(scaler, lda)``。

    ``scaler`` 与 ``lda`` 必须成对使用：判别方向定义在标准化后的空间里，
    对训练/测试特征施加投影时要用同一个 ``scaler``。
    """
    run_groups = np.asarray(run_groups)
    n_runs = len(np.unique(run_groups))
    if n_runs < 2:
        raise ValueError(f"run 分类器至少需要 2 个 run，实际 {n_runs} 个")

    scaler = StandardScaler()
    x = scaler.fit_transform(feats)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(x, run_groups)
    return scaler, lda


def directions_from_lda(lda: LinearDiscriminantAnalysis, k: int) -> tuple[np.ndarray, int, int]:
    """取判别力最强的 ``k`` 个方向并正交化。

    Returns:
        ``(directions, k_effective, n_available)``：``directions`` 是
        ``(特征数, k_effective)`` 的正交基；``n_available`` 是该 run 分类器
        可提供的判别方向总数（训练 run 数 − 1），用于判断 ``k`` 是否被截断。
    """
    scalings = np.asarray(lda.scalings_)
    ratios = np.asarray(getattr(lda, "explained_variance_ratio_", np.array([])))
    if ratios.size > 0:
        # solver="eigen" 下 scalings_ 是方阵，只有前 n_classes-1 列对应非零判别方向
        n_available = int(ratios.size)
        basis = scalings[:, :n_available]
    else:
        n_available = int(scalings.shape[1])
        basis = scalings

    k_effective = int(min(max(k, 0), n_available))
    if k_effective == 0:
        return np.zeros((scalings.shape[0], 0), dtype=scalings.dtype), 0, n_available

    if np.any(ratios > 0):
        order = np.argsort(ratios)[::-1][:k_effective]
    else:  # 退化兜底：没有可解释方差时按原顺序取
        order = np.arange(k_effective)
    selected = basis[:, order]

    # LDA 的判别方向彼此不正交，先 SVD 取一组正交基再投影
    u, _, _ = np.linalg.svd(selected, full_matrices=False)
    return u[:, :k_effective], k_effective, n_available


def project(feats: np.ndarray, scaler: StandardScaler, directions: np.ndarray) -> np.ndarray:
    """标准化后投影到 ``directions`` 的正交补空间。"""
    x = scaler.transform(feats)
    if directions.shape[1] == 0:
        return x
    return x - (x @ directions) @ directions.T
