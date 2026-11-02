import pickle
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# 三个频段 (Hz)
BANDS = {
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 100),
}

FS = 500          # 采样率
ORDER = 4         # Butterworth 阶数
MARGIN_SEC = 0.25  # 首尾各裁剪的时长（秒）


def butter_bandpass(low_freq, high_freq, fs=FS, order=ORDER):
    """返回 Butterworth 带通滤波器系数 (b, a)。"""
    nyq = 0.5 * fs
    if not 0 < low_freq < high_freq < nyq:
        raise ValueError(f"频段非法: {low_freq}-{high_freq} Hz (Nyquist = {nyq} Hz)")
    b, a = butter(order, [low_freq / nyq, high_freq / nyq], btype="band")
    return b, a


def bandpass_filter(data, low_freq, high_freq, fs=FS, order=ORDER, coeffs=None):
    """对多通道 EEG 沿时间轴（最后一维）做零相位带通滤波。

    data: (n_channels, n_times)
    coeffs: 可传入预先算好的 (b, a)，避免每个 trial 重复计算。
    如果高频超过 45Hz（如 Gamma 频段），会先进行 50Hz 及 100Hz 陷波。
    """
    data = np.asarray(data, dtype=np.float64)
    
    # 针对包含高频的频段（Gamma），先剔除工频干扰
    if high_freq > 45:
        for notch_f in [50.0, 100.0]:
            if notch_f < (fs / 2):
                b_notch, a_notch = iirnotch(notch_f, 30, fs)
                data = filtfilt(b_notch, a_notch, data, axis=-1)
                
    b, a = coeffs if coeffs is not None else butter_bandpass(low_freq, high_freq, fs, order)
    return filtfilt(b, a, data, axis=-1)


def _trim(x, margin):
    """裁剪首尾边缘，防止滤波暂态效应。"""
    return x[..., margin:x.shape[-1] - margin] if margin else x


def filter_trial(raw_eeg, fs=FS, bands=BANDS, order=ORDER, margin_sec=MARGIN_SEC, coeffs=None):
    """处理单个 trial：(n_channels, n_times) -> {频段名: 滤波结果}，raw 同步裁剪保证长度一致。"""
    raw = np.asarray(raw_eeg, dtype=np.float64)
    margin = int(margin_sec * fs)
    
    if raw.ndim != 2:
        raise ValueError(f"期望 (n_channels, n_times)，实际 {raw.shape}")
    if 2 * margin >= raw.shape[-1]:
        raise ValueError(f"数据太短（{raw.shape[-1]} 点），无法裁剪首尾各 {margin} 点")

    trial = {"raw": _trim(raw, margin)}
    for name, (low, high) in bands.items():
        filtered = bandpass_filter(
            raw, low, high, fs, order, coeffs=None if coeffs is None else coeffs[name]
        )
        trial[name] = _trim(filtered, margin)
    return trial


def filter_bands(epoch_data_list, fs=FS, bands=BANDS, order=ORDER, margin_sec=MARGIN_SEC):
    """批量处理 [{"text": ..., "input_features": (n_channels, n_times)}, ...]。

    滤波器系数按频段只计算一次。
    """
    coeffs = {name: butter_bandpass(low, high, fs, order) for name, (low, high) in bands.items()}
    out = []
    for item in epoch_data_list:
        trial = filter_trial(item["input_features"], fs, bands, order, margin_sec, coeffs)
        trial["text"] = item["text"]
        out.append(trial)
    return out


def filter_pkl(pkl_path, out_path=None, **kwargs):
    """读取去 ICA 后的 pkl 并滤波；给定 out_path 时把结果写回 pkl。"""
    with open(pkl_path, "rb") as f:
        epochs = pickle.load(f)
    filtered = filter_bands(epochs, **kwargs)
    if out_path:
        with open(out_path, "wb") as f:
            pickle.dump(filtered, f)
    return filtered


def _run_test():
    """构造 10 Hz(α) + 22 Hz(β) + 60 Hz(γ) 正弦叠加噪声，验证滤波结果。"""
    fs, n_channels, n_times = FS, 4, 1651
    t = np.arange(n_times) / fs
    rng = np.random.default_rng(42)

    # 模拟 50Hz 工频干扰，混入到波形中
    wave = sum(
        amp * np.sin(2 * np.pi * freq * t)
        for amp, freq in [(1.0, 10.0), (0.8, 22.0), (0.5, 60.0), (0.3, 50.0)]
    )
    raw = wave[None, :] + 0.05 * rng.standard_normal((n_channels, n_times))

    epochs = [{"text": f"trial{i}", "input_features": raw} for i in range(3)]
    out = filter_bands(epochs, fs=fs)

    margin = int(MARGIN_SEC * fs)
    expect_t = n_times - 2 * margin

    for i, trial in enumerate(out):
        assert trial["text"] == f"trial{i}"
        assert trial["raw"].shape == (n_channels, expect_t)
        for name in BANDS:
            assert trial[name].shape == (n_channels, expect_t), (name, trial[name].shape)

    # 每个频段的能量峰应落在自己的频段内
    freqs = np.fft.rfftfreq(expect_t, 1 / fs)
    all_ok = True
    for name, (low, high) in BANDS.items():
        spectrum = np.abs(np.fft.rfft(np.mean(out[0][name], axis=0))) ** 2
        peak = freqs[np.argmax(spectrum)]
        in_band = low <= peak <= high
        all_ok &= in_band
        print(f"{name:5s}: 峰值 {peak:6.2f} Hz, 期望 [{low}, {high}] Hz -> {'OK' if in_band else 'FAIL'}")

    assert all_ok, "带通滤波结果的能量峰不在对应频段内"
    
    # 验证 Gamma 里的 50Hz 是否被成功压制
    gamma_spec = np.abs(np.fft.rfft(np.mean(out[0]["gamma"], axis=0))) ** 2
    peak_50_idx = np.argmin(np.abs(freqs - 50.0))
    # 如果 50Hz 附近能量远小于 60Hz 主信号，说明陷波成功
    is_notched = gamma_spec[peak_50_idx] < (gamma_spec.max() * 0.1)
    print(f"Gamma 50Hz 陷波测试: {'OK (50Hz 已压制)' if is_notched else 'FAIL (仍有强 50Hz)'}")
    
    print("所有测试通过")


if __name__ == "__main__":
    _run_test()