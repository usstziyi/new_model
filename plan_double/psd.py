"""频域分支的 EEG 功率谱密度（PSD）特征提取。

论文 3.2.2 用 Welch 法估计功率谱，再按 δ/θ/α/β/γ 五个频段取功率。本模块输出
``(C, 5)``（或 ``(B, C, 5)``），可直接作为 ``model.TimeFreqDualBranchNet`` 的
``x_freq_psd`` 参数。

实现只依赖本环境已有的 ``numpy 2.5.4`` + ``scipy 1.18.1``（``scipy.signal.welch``），
不需要 mne 的 Epochs 对象。

典型用法::

    from psd import eeg_band_psd, eeg_band_psd_batch

    x_psd = eeg_band_psd(eeg, fs=500)          # eeg (C, T)   -> (C, 5)
    x_psd = eeg_band_psd_batch(eeg, fs=500)    # eeg (B, C, T) -> (B, C, 5)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirnotch, welch

# 采样率（与 plan_single_cnn_mlp/band.py 的 FS 一致）
FS = 500.0

# Welch 窗长：1 s -> 频率分辨率 1 Hz；T 较短时会自动缩短
NPERSEG = 500

# 五个频段（Hz），顺序即输出最后一维的顺序
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "gamma": (30.0, 70.0),
}

# 工频陷波点：50 Hz 落在 gamma(30–70) 内，不陷波会污染 gamma 功率
NOTCH_FREQS = (50.0, 100.0)


def notch_filter(eeg, fs: float = FS, freqs=NOTCH_FREQS, q: float = 30.0):
    """对 ``(..., T)`` 的 EEG 做零相位工频陷波，返回新数组。"""
    data = np.asarray(eeg, dtype=np.float64)
    for f in freqs:
        if f < fs / 2:
            b, a = iirnotch(f, q, fs)
            data = filtfilt(b, a, data, axis=-1)
    return data


def welch_psd(eeg, fs: float = FS, nperseg: int = NPERSEG, notch: bool = True):
    """Welch 功率谱密度。

    Args:
        eeg: ``(..., T)``，单位不限（µV 或 V），沿最后一维为时间。
        fs: 采样率。
        nperseg: 每段长度，会被截断到不超过 ``T``。
        notch: 是否先做 50/100 Hz 零相位陷波。

    Returns:
        ``(freqs, psd)``，``freqs`` 为 ``(F,)``，``psd`` 为 ``(..., F)``，
        形状比输入少一维时间轴。
    """
    x = notch_filter(eeg, fs) if notch else np.asarray(eeg, dtype=np.float64)
    nperseg = int(min(nperseg, x.shape[-1]))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, detrend="constant", axis=-1)
    return freqs, psd


def band_power(freqs, psd, bands: dict[str, tuple[float, float]] = BANDS,
               mode: str = "mean"):
    """按频段汇总 PSD。

    Args:
        freqs: ``(F,)``，``welch_psd`` 返回的频率轴。
        psd: ``(..., F)``。
        mode: ``"mean"`` 取频段内平均功率（默认）；``"integrate"`` 对频率积分。

    Returns:
        ``(..., len(bands))``，最后一维顺序与 ``bands`` 一致。
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    out = []
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if not mask.any():
            raise ValueError(
                f"频段 {name} ({low}-{high} Hz) 内没有频点，"
                f"有效分辨率 {freqs[1] - freqs[0]:.2f} Hz"
            )
        if mode == "mean":
            out.append(psd[..., mask].mean(axis=-1))
        elif mode == "integrate":
            out.append(np.trapezoid(psd[..., mask], freqs[mask], axis=-1))
        else:
            raise ValueError(f"mode 需为 mean/integrate，收到 {mode!r}")
    return np.stack(out, axis=-1)


def eeg_band_psd(eeg, fs: float = FS, bands: dict[str, tuple[float, float]] = BANDS,
                 nperseg: int = NPERSEG, notch: bool = True,
                 log: bool = False, relative: bool = False) -> np.ndarray:
    """``(..., T)`` 的 EEG -> ``(..., len(bands))`` 频段功率。

    Args:
        log: 是否取 ``log10``，压缩跨频段的量级差异（gamma 功率通常远小于 delta）。
        relative: 是否转成相对功率（各频段之和为 1），可抵消通道间幅度差异。
    """
    freqs, psd = welch_psd(eeg, fs=fs, nperseg=nperseg, notch=notch)
    power = band_power(freqs, psd, bands)

    if relative:
        power = power / np.maximum(power.sum(axis=-1, keepdims=True), 1e-30)
    if log:
        power = np.log10(np.maximum(power, 1e-30))
    return power.astype(np.float32)


def eeg_band_psd_batch(eeg, **kwargs) -> np.ndarray:
    """``(B, C, T)`` -> ``(B, C, len(bands))``，参数同 :func:`eeg_band_psd`。"""
    return eeg_band_psd(np.asarray(eeg), **kwargs)


def _run_test() -> None:
    """合成 α/β/γ 正弦 + 50 Hz 工频 + 噪声，检查功率落在正确频段。"""
    fs, n_channels, n_times = FS, 3, 1401
    t = np.arange(n_times) / fs
    rng = np.random.default_rng(42)

    wave = sum(
        amp * np.sin(2 * np.pi * freq * t)
        for amp, freq in [(1.0, 10.0), (0.7, 22.0), (0.4, 60.0), (0.6, 50.0)]
    )
    raw = wave[None, :] + 0.05 * rng.standard_normal((n_channels, n_times))

    power = eeg_band_psd(raw, fs=fs)
    assert power.shape == (n_channels, len(BANDS)), power.shape
    print(f"单试次: (C, T)=({n_channels}, {n_times}) -> {power.shape}")

    # 每段的主成分应落在构造它的频段上
    dominant = np.argmax(power, axis=-1)
    expected = np.array([list(BANDS).index(name) for name in ("alpha",)], dtype=np.int64)
    ok_alpha = bool((dominant == expected[0]).all())
    print(f"alpha 最强: {'OK' if ok_alpha else 'FAIL'}，各通道最强频段 {list(BANDS)[dominant[0]]}")

    # gamma 里含 50 Hz 工频，陷波后应仍由 60 Hz 主导
    no_notch = eeg_band_psd(raw, fs=fs, notch=False)
    g = list(BANDS).index("gamma")
    ratio = power[:, g].mean() / no_notch[:, g].mean()
    print(f"陷波后 gamma 功率比: {ratio:.3f} ({'OK' if ratio < 0.9 else 'FAIL'})")

    # 相对功率与 log
    rel = eeg_band_psd(raw, fs=fs, relative=True)
    assert np.allclose(rel.sum(axis=-1), 1.0, atol=1e-5)
    logp = eeg_band_psd(raw, fs=fs, log=True)
    print(f"相对功率和为 1: OK；log 输出范围 [{logp.min():.2f}, {logp.max():.2f}]")

    # batch 形状
    batch = np.stack([raw, raw * 2.0])
    out = eeg_band_psd_batch(batch, fs=fs)
    assert out.shape == (2, n_channels, len(BANDS)), out.shape
    print(f"批量: {batch.shape} -> {out.shape}")

    assert ok_alpha, "alpha 频段未成为主成分"
    print("所有测试通过")


if __name__ == "__main__":
    _run_test()
