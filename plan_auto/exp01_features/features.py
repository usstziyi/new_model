"""实验 01 的特征提取。

所有特征只依赖单试次信号，不跨试次使用统计量；标准化 / 协方差均值等
需要“训练集统计量”的步骤放在交叉验证内部，避免泄漏。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402

BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("lowgamma", 30.0, 45.0),
)
FEATURE_ROOT = PROJECT_ROOT / "plan_auto/cache/features"


def _psd_bandpower(x: np.ndarray, sfreq: float, nperseg: int) -> np.ndarray:
    """x: (N, C, T) -> (N, C, n_bands) 的 log10 平均功率。"""
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    out = np.empty((x.shape[0], x.shape[1], len(BANDS)), dtype=np.float32)
    for i, (_, low, high) in enumerate(BANDS):
        mask = (freqs >= low) & (freqs < high)
        out[:, :, i] = np.log10(np.maximum(psd[:, :, mask].mean(axis=-1), 1e-12))
    return out


def bandpower_all(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """全时段频带功率，返回 (N, C*5)。"""
    feats = _psd_bandpower(eeg, sfreq, nperseg=256)
    return feats.reshape(len(eeg), -1)


def bandpower_windows(eeg: np.ndarray, sfreq: float, n_windows: int = 4) -> np.ndarray:
    """把 3.3 s 均分为 n_windows 段，每段算频带功率，返回 (N, n_windows*C*5)。"""
    edges = np.linspace(0, eeg.shape[-1], n_windows + 1).astype(int)
    parts = [_psd_bandpower(eeg[..., edges[i]:edges[i + 1]], sfreq, nperseg=128) for i in range(n_windows)]
    feats = np.concatenate(parts, axis=-1)  # (N, C, n_windows*5)
    return feats.reshape(len(eeg), -1)


def erp_features(eeg: np.ndarray, sfreq: float, target_hz: float = 20.0) -> np.ndarray:
    """降采样到 target_hz 的时域波形，返回 (N, C*T')，用于捕捉诱发响应。"""
    step = max(int(round(sfreq / target_hz)), 1)
    return np.ascontiguousarray(eeg[:, :, ::step], dtype=np.float32).reshape(len(eeg), -1)


def hjorth_features(eeg: np.ndarray) -> np.ndarray:
    """Hjorth activity / mobility / complexity，返回 (N, C*3)。"""
    x = np.asarray(eeg, dtype=np.float32)
    d1 = np.diff(x, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0 = x.var(axis=-1)
    var1 = d1.var(axis=-1)
    var2 = d2.var(axis=-1)
    mobility = np.sqrt(np.divide(var1, np.maximum(var0, 1e-12)))
    mobility_d1 = np.sqrt(np.divide(var2, np.maximum(var1, 1e-12)))
    complexity = np.divide(mobility_d1, np.maximum(mobility, 1e-12))
    feats = np.stack([np.log10(np.maximum(var0, 1e-12)), mobility, complexity], axis=-1)
    return feats.reshape(len(x), -1)


def build_features(subject: str, overwrite: bool = False) -> dict[str, np.ndarray]:
    """计算并缓存该被试的全部静态特征。"""
    out_dir = FEATURE_ROOT / subject
    path = out_dir / "features.npz"
    if path.exists() and not overwrite:
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}

    cache = D.load_cache(subject)
    eeg = np.asarray(cache["eeg"], dtype=np.float32)
    sfreq = float(cache["meta"]["target_sfreq"])
    ref = eeg
    features = {
        "bp": bandpower_all(ref, sfreq),
        "bpwin": bandpower_windows(ref, sfreq, n_windows=4),
        "erp": erp_features(ref, sfreq),
        "hjorth": hjorth_features(ref),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)
    print(f"[Features] {subject} 已缓存 {', '.join(f'{k}{v.shape}' for k, v in features.items())}", flush=True)
    return features


# ------------------------------------------------------------ 协方差切空间


def _covariances(x: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """x: (N, C, T) -> 正则化协方差 (N, C, C)。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", x, x) / max(x.shape[-1] - 1, 1)
    scale = np.trace(cov, axis1=1, axis2=2)[:, None, None] / cov.shape[1]
    return cov + reg * scale * np.eye(cov.shape[1])[None]


def _riemann_mean(covs: np.ndarray, n_iter: int = 5) -> np.ndarray:
    """协方差矩阵的几何均值（迭代 log-Euclidean 近似）。"""
    mean = covs.mean(axis=0)
    for _ in range(n_iter):
        vals, vecs = np.linalg.eigh(mean)
        vals = np.maximum(vals, 1e-12)
        inv_sqrt = vecs @ np.diag(vals ** -0.5) @ vecs.T
        sqrt = vecs @ np.diag(vals ** 0.5) @ vecs.T
        inner = inv_sqrt[None] @ covs @ inv_sqrt[None]
        vals2, vecs2 = np.linalg.eigh(inner)
        log_eig = np.log(np.maximum(vals2, 1e-12))[..., None, :]
        log_inner = (vecs2 * log_eig) @ np.swapaxes(vecs2, -1, -2)
        mean = sqrt @ np.exp(log_inner.mean(axis=0)) @ sqrt
    return mean


def tangent_transform(
    train_x: np.ndarray, other_x: list[np.ndarray], reg: float = 1e-6
) -> tuple[np.ndarray, list[np.ndarray]]:
    """用训练集几何均值把协方差投影到切空间（上三角向量）。"""
    train_cov = _covariances(train_x, reg)
    mean = _riemann_mean(train_cov)
    vals, vecs = np.linalg.eigh(mean)
    inv_sqrt = vecs @ np.diag(np.maximum(vals, 1e-12) ** -0.5) @ vecs.T
    tri = np.triu_indices(mean.shape[0])

    def project(covs: np.ndarray) -> np.ndarray:
        whitened = inv_sqrt[None] @ covs @ inv_sqrt[None]
        vals2, vecs2 = np.linalg.eigh(whitened)
        log_eig = np.log(np.maximum(vals2, 1e-12))[..., None, :]
        logs = (vecs2 * log_eig) @ np.swapaxes(vecs2, -1, -2)
        out = logs[:, tri[0], tri[1]]
        out *= np.sqrt(2.0)
        diag = tri[0] == tri[1]
        out[:, diag] /= np.sqrt(2.0)
        return out.astype(np.float32)

    return project(train_cov), [project(_covariances(x, reg)) for x in other_x]
