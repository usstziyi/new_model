"""验证 band.py 的 α/β/γ 频段分解是否准确。

从 ``derivatives/preprocessed_pkl`` 里抽取某个 run 的若干 trial，用频谱指标检查
``band.filter_trial`` 的输出：

- 每个频段的谱峰是否落在自己的频段内；
- 通带增益是否接近 0 dB（滤波没有把带内信号整体削掉）；
- 带内能量占比、边沿 / 远端泄漏；
- 与 8 阶 Butterworth 参考滤波的波形相关系数（衡量与“理想频段”的接近程度）；
- α/β/γ 之间的相关系数（应接近 0，说明频段彼此独立）；
- 50 Hz 工频陷波是否把原谱线压到 EEG 底噪以下。

用法（在项目根目录）::

    # 默认取 sub-01 imagine run-01 的首 / 中 / 尾三个 trial
    uv run scripts/check_band_decomposition.py

    # 指定被试、任务、run 与 trial 下标
    uv run scripts/check_band_decomposition.py --subject sub-02 --task reading --run 01 --trials 0 50 180

    # 同时把结果写成 CSV；省略路径时自动存到 outputs/reports/
    uv run scripts/check_band_decomposition.py --report
    uv run scripts/check_band_decomposition.py --report my_band_check.csv

有任何一项不达标时脚本以退出码 1 结束，便于批处理时发现问题。
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

REPO_ROOT = Path(__file__).resolve().parents[1]
PKL_ROOT = REPO_ROOT / "datasets/ds005170-download/derivatives/preprocessed_pkl"
BAND_DIR = REPO_ROOT / "plan_csfa_pmccnn"

sys.path.insert(0, str(BAND_DIR))
from band import BANDS, FS, filter_trial  # noqa: E402

N_EEG_CHANNELS = 122
REF_ORDER = 8           # 参考滤波阶数（比默认 4 阶更陡，用来衡量“理想频段”）
PSD_NPERSEG = 500       # 500 点 @ 500 Hz -> 1 Hz 整数频点
LINE_FREQ_RANGE = (48.0, 52.0)

# CSV 里 task 叫 reading，pkl 文件名里叫 read
TASK_ALIASES = {"reading": "read"}

# --report 不带路径时写到这里
DEFAULT_REPORT_DIR = REPO_ROOT / "outputs" / "reports"

CSV_FIELDS = [
    "subject", "task", "run", "trial", "text", "band", "low_hz", "high_hz",
    "raw_len", "trimmed_len", "margin",
    "peak_hz", "peak_ok", "gain_db", "gain_ok",
    "in_band_pct", "edge_low_pct", "edge_high_pct", "far_low_pct", "far_high_pct", "purity_ok",
    "ref_corr", "ref_rms_pct", "ref_ok",
    "corr_alpha_beta", "corr_alpha_gamma", "corr_beta_gamma",
    "line_freq_hz", "line_suppression_db", "line_residual_over_floor", "notch_ok",
    "dc_max_v", "length_ok", "band_pass", "trial_pass",
]


def find_pkl(subject: str, task: str, run: int) -> Path:
    """按 subject / task / run 定位 pkl（容忍 run 文件名零填充差异）。"""
    task = TASK_ALIASES.get(task, task)
    eeg_dir = PKL_ROOT / subject / "eeg"
    matches = [
        path
        for path in sorted(eeg_dir.glob(f"{subject}_task-{task}_run-*_eeg.pkl"))
        if int(path.stem.split("_run-")[1].split("_")[0]) == run
    ]
    if not matches:
        raise FileNotFoundError(
            f"未找到 pkl: subject={subject}, task={task}, run={run}（目录 {eeg_dir}）"
        )
    if len(matches) > 1:
        raise RuntimeError("匹配到多个 pkl:\n" + "\n".join(str(p) for p in matches))
    return matches[0]


def psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """通道平均 Welch 功率谱，nperseg 取 500 点让频点为整数 Hz。"""
    nperseg = min(PSD_NPERSEG, x.shape[-1])
    freqs, power = welch(
        x, fs=FS, nperseg=nperseg, noverlap=nperseg // 2, axis=-1
    )
    return freqs, power.mean(axis=0)


def band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapezoid(power[mask], freqs[mask]))


def reference_band(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """参考实现：与 band.py 相同的陷波，再叠加 8 阶 SOS 带通。"""
    if high > 45:
        for notch_freq in (50.0, 100.0):
            if notch_freq < FS / 2:
                b, a = iirnotch(notch_freq, 30, FS)
                x = filtfilt(b, a, x, axis=-1)
    sos = butter(REF_ORDER, [low / (FS / 2), high / (FS / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def fine_spectrum(x: np.ndarray, zero_pad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """补零 FFT 得到细粒度幅度谱（通道平均），用于精确定位 50 Hz 谱线。"""
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    spectrum = np.abs(
        np.fft.rfft(x * np.hanning(n), n=n * zero_pad, axis=-1)
    ).mean(axis=0)
    return np.fft.rfftfreq(n * zero_pad, 1 / FS), spectrum


def analyze_trial(
    raw: np.ndarray, subject: str, task: str, run: int, trial_index: int, text: str
) -> tuple[bool, list[dict]]:
    """对单个 trial 做完整检查，打印报告并返回 (是否全部达标, CSV 行列表)。"""
    out = filter_trial(raw)
    raw_trim = out["raw"]
    margin = (raw.shape[-1] - raw_trim.shape[-1]) // 2
    freqs, power_raw = psd(raw_trim)

    print(f"\n=== trial #{trial_index} {text[:20]!r} ===")
    print(f"  形状 {raw.shape} -> {raw_trim.shape}（首尾各裁 {margin} 点）")

    ok = True
    band_rows: list[dict] = []
    dc: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        signal = out[name]
        _, power = psd(signal)

        peak = float(freqs[np.argmax(power)])
        peak_ok = low <= peak <= high

        # 通带增益：带内去掉两端 1.5 Hz 过渡区后，滤波后/原始 PSD 的中位比值
        core = (freqs >= low + 1.5) & (freqs <= high - 1.5)
        gain_db = float(
            10 * np.log10(np.median(power[core] / np.maximum(power_raw[core], 1e-30)))
        )
        gain_ok = abs(gain_db) < 1.0

        # 能量分布：带内 / 边沿 1 Hz / 更远
        total = band_energy(freqs, power, 0.0, FS / 2)
        in_band = band_energy(freqs, power, low, high) / total
        edge_low = band_energy(freqs, power, max(low - 1, 0), low) / total
        edge_high = band_energy(freqs, power, high, min(high + 1, FS / 2)) / total
        far_low = band_energy(freqs, power, 0.0, max(low - 1, 0)) / total
        far_high = band_energy(freqs, power, min(high + 1, FS / 2), FS / 2) / total
        purity_ok = in_band >= 0.85

        # 与 8 阶参考的波形吻合度
        ref = reference_band(raw, low, high)[:, margin:margin + raw_trim.shape[-1]]
        corr = float(np.corrcoef(signal.ravel(), ref.ravel())[0, 1])
        rms_rel = float(
            np.sqrt(np.mean((signal - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))
        )
        corr_ok = corr > 0.95

        band_ok = peak_ok and gain_ok and purity_ok and corr_ok
        ok &= band_ok
        dc[name] = float(np.abs(signal.mean(axis=-1)).max())
        band_rows.append(
            {
                "band": name,
                "low_hz": low,
                "high_hz": high,
                "raw_len": raw.shape[-1],
                "trimmed_len": raw_trim.shape[-1],
                "margin": margin,
                "peak_hz": round(peak, 2),
                "peak_ok": int(peak_ok),
                "gain_db": round(gain_db, 3),
                "gain_ok": int(gain_ok),
                "in_band_pct": round(in_band * 100, 2),
                "edge_low_pct": round(edge_low * 100, 2),
                "edge_high_pct": round(edge_high * 100, 2),
                "far_low_pct": round(far_low * 100, 2),
                "far_high_pct": round(far_high * 100, 2),
                "purity_ok": int(purity_ok),
                "ref_corr": round(corr, 4),
                "ref_rms_pct": round(rms_rel * 100, 2),
                "ref_ok": int(corr_ok),
                "dc_max_v": float(dc[name]),
                "band_pass": int(band_ok),
            }
        )
        print(
            f"  {name:5s} 峰值 {peak:5.1f} Hz {'OK' if peak_ok else 'FAIL'} | "
            f"通带增益 {gain_db:+5.2f} dB {'OK' if gain_ok else 'FAIL'} | "
            f"带内 {in_band * 100:4.1f}%（边沿 {edge_low * 100:3.1f}/{edge_high * 100:3.1f}%，"
            f"远端 {far_low * 100:3.1f}/{far_high * 100:3.1f}%）{'OK' if purity_ok else 'FAIL'} | "
            f"与{REF_ORDER}阶参考 corr {corr:.4f} / RMS差 {rms_rel * 100:4.1f}% "
            f"{'OK' if corr_ok else 'FAIL'}"
        )

    # 频段独立性
    pairs = (("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma"))
    corrs = {
        f"{a}-{b}": float(np.corrcoef(out[a].ravel(), out[b].ravel())[0, 1])
        for a, b in pairs
    }
    print("  频段间相关: " + "  ".join(f"{k}={v:+.3f}" for k, v in corrs.items()))

    # 50 Hz 工频：定位原始谱线，再看 gamma 分支是否压到 EEG 底噪以下
    freqs_fine, spec_raw = fine_spectrum(raw_trim)
    _, spec_gamma = fine_spectrum(out["gamma"])
    sel = (freqs_fine >= LINE_FREQ_RANGE[0]) & (freqs_fine <= LINE_FREQ_RANGE[1])
    line_freq = float(freqs_fine[sel][np.argmax(spec_raw[sel])])
    i_line = int(np.argmin(np.abs(freqs_fine - line_freq)))
    amp_raw = float(spec_raw[i_line])
    amp_gamma = float(spec_gamma[i_line])
    suppression_db = float(
        20 * np.log10(max(amp_gamma, 1e-30) / max(amp_raw, 1e-30))
    )
    # 底噪：同一 45-55 Hz 区间内、谱线 ±1.5 Hz 之外的中位数
    noise_mask = sel & (np.abs(freqs_fine - line_freq) > 1.5)
    floor = float(np.median(spec_gamma[noise_mask]))
    noise_ratio = amp_gamma / max(floor, 1e-30)
    notch_ok = suppression_db < -15 and amp_gamma <= floor
    ok &= notch_ok
    print(
        f"  工频 {line_freq:.2f} Hz：原始 -> gamma 幅度 {suppression_db:6.1f} dB，"
        f"残值/底噪 {noise_ratio:4.2f} "
        f"{'OK（已压到底噪以下）' if notch_ok else 'FAIL'}"
    )

    # 裁剪与直流
    lengths = {name: out[name].shape[-1] for name in ("raw", *BANDS)}
    length_ok = len(set(lengths.values())) == 1
    ok &= length_ok
    print(
        f"  长度一致 {'OK' if length_ok else 'FAIL'} {lengths} | "
        + ", ".join(f"{n} 直流 {v:.1e} V" for n, v in dc.items())
    )

    trial_common = {
        "subject": subject,
        "task": task,
        "run": run,
        "trial": trial_index,
        "text": text,
        "corr_alpha_beta": round(corrs["alpha-beta"], 4),
        "corr_alpha_gamma": round(corrs["alpha-gamma"], 4),
        "corr_beta_gamma": round(corrs["beta-gamma"], 4),
        "line_freq_hz": round(line_freq, 2),
        "line_suppression_db": round(suppression_db, 2),
        "line_residual_over_floor": round(noise_ratio, 3),
        "notch_ok": int(notch_ok),
        "length_ok": int(length_ok),
        "trial_pass": int(ok),
    }
    for row in band_rows:
        row.update(trial_common)
    return ok, band_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 band.py 频段分解的准确性。")
    parser.add_argument("--subject", default="sub-01", help="被试，如 sub-01")
    parser.add_argument("--task", default="imagine", help="任务，imagine / read，默认 imagine")
    parser.add_argument("--run", type=int, default=1, help="run 编号，默认 1")
    parser.add_argument(
        "--trials", type=int, nargs="+", default=None,
        help="试次下标；默认取首 / 中 / 尾三个",
    )
    parser.add_argument(
        "--report", type=Path, nargs="?", const=Path("auto"), default=None, metavar="CSV",
        help="把结果写成 CSV；不带路径时存到 outputs/reports/ 下自动命名",
    )
    args = parser.parse_args()

    pkl_path = find_pkl(args.subject, args.task, args.run)
    with pkl_path.open("rb") as handle:
        items = pickle.load(handle)
    if not items:
        raise RuntimeError(f"pkl 为空: {pkl_path}")

    trials = args.trials or [0, len(items) // 2, len(items) - 1]
    print(f"######## {pkl_path.name}：{len(items)} trials，检查下标 {trials} ########")

    all_ok = True
    rows: list[dict] = []
    for index in trials:
        features = np.asarray(items[index]["input_features"])
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        eeg = features[:N_EEG_CHANNELS]
        text = str(items[index].get("text", ""))
        trial_ok, band_rows = analyze_trial(
            eeg, args.subject, args.task, args.run, index, text
        )
        all_ok &= trial_ok
        rows.extend(band_rows)

    print(f"\n总体: {'PASS' if all_ok else 'FAIL'}")

    if args.report is not None:
        if args.report == Path("auto"):
            report_path = DEFAULT_REPORT_DIR / (
                f"band_decomposition_{args.subject}"
                f"_task-{TASK_ALIASES.get(args.task, args.task)}_run-{args.run:02d}.csv"
            )
        else:
            report_path = args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Report] {report_path}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
