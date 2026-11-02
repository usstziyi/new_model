"""绘制某个 EDF 的辅助通道（EKG / EMG / VEO / HEO）波形，用于肉眼检查。

架构与 plot_edf_waveform.py 一致：按 subject / ses / task / run + 时间窗定位 EDF，
只读取窗口内数据，并把窗口内的试次起点（Trigger 事件 65380）用蓝色虚线标出。

不同点：这四个通道幅度量级差别很大（EEG 以 µV 计，EMG 峰峰值可达数十 V），
叠在一张图里会被撑爆。所以每个通道单独占一个子图、各自独立缩放（单位 mV，
纵轴按 99.5 分位截断，超出部分会被裁掉），横向共享同一条时间轴。

用法::

    # 看 run-01 开头 20 秒
    uv run scripts/plot_aux_waveform.py --subject sub-01 --ses 01 --run 01 --start 0 --end 20

    # 看第一个试次附近（run-01 首个 65380 事件在 89.5 s）
    uv run scripts/plot_aux_waveform.py --subject 01 --ses ses-01 --run 010 --start 85 --end 105

    # 只看心电和肌电
    uv run scripts/plot_aux_waveform.py --subject sub-01 --ses 01 --run 01 --start 85 --end 105 --channels EKG EMG

    # 看整段（点数会自动抽稀）
    uv run scripts/plot_aux_waveform.py --subject sub-01 --ses 01 --run 045

默认画四个辅助通道中实际存在的那些；图保存在
``figures/aux_waveform_<subject>_<ses>_task-<task>_run-<run>_<start>-<end>s.png``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只存图，不依赖图形界面

import matplotlib.pyplot as plt
import numpy as np

# 文件定位 / 窗口读取等通用逻辑直接复用 waveform 脚本，避免两份实现漂移
from plot_edf_waveform import (
    DEFAULT_OUT_DIR,
    TRIAL_DESCRIPTION,
    find_edf,
    load_segment,
    normalize_run,
    normalize_session,
    normalize_subject,
    run_label,
)

# 要绘制的四个辅助通道（EDF 里标为 eeg，但不是脑电）
AUX_CHANNELS = ("EKG", "EMG", "VEO", "HEO")

# 每通道最多画这么多点，超出就按步长抽稀（只影响显示，不改数据）
MAX_POINTS_PER_CHANNEL = 4000

# 纵轴按这个分位数截断，避免个别大尖峰把整条曲线压成一条线
YLIM_PERCENTILE = 99.5


def pick_unit(max_abs_mv: float) -> tuple[float, str]:
    """按最大幅度挑一个统一的显示单位（四个子图共用），返回 (换算系数, 单位)。"""
    if max_abs_mv >= 1e3:   # >= 1 V
        return 1e-3, "V"
    if max_abs_mv >= 1.0:   # >= 1 mV
        return 1.0, "mV"
    if max_abs_mv >= 1e-3:  # >= 1 µV
        return 1e3, "µV"
    return 1e6, "nV"


def resolve_channels(raw, requested: list[str] | None) -> list[str]:
    """返回文件里实际存在的待绘通道（保持 AUX_CHANNELS 的固定顺序）。"""
    wanted = [ch.upper() for ch in requested] if requested else list(AUX_CHANNELS)
    names = {ch.upper(): ch for ch in raw.ch_names}

    channels, missing = [], []
    for name in wanted:
        if name in names:
            channels.append(names[name])
        else:
            missing.append(name)
    if missing:
        print(f"[Warn]   channels not found in file: {missing}")
    if not channels:
        raise RuntimeError(f"文件里没有这些通道: {wanted} (现有 {raw.ch_names})")
    return channels


def plot_aux(
    raw,
    channels: list[str],
    out_path: Path,
    dpi: int,
    title: str,
    window_start: float,
    event_onsets: np.ndarray,
) -> None:
    """每个辅助通道一个子图，纵向堆叠、各自缩放，单位按量级自动选择。"""
    data = raw.get_data(picks=channels) * 1e3  # V -> mV
    # crop 后 raw.times 从 0 开始，加回窗口起点还原绝对时间
    times = raw.times + window_start
    decim = max(1, int(np.ceil(data.shape[1] / MAX_POINTS_PER_CHANNEL)))
    data = data[:, ::decim]
    times = times[::decim]

    data = data - np.median(data, axis=1, keepdims=True)

    # 先按 mV 定每条通道的显示范围，再据此挑一个统一单位，避免刻度出现 40000 这种大数
    limits = np.zeros(len(channels))
    for i in range(len(channels)):
        limit = float(np.percentile(np.abs(data[i]), YLIM_PERCENTILE)) * 1.1
        if not np.isfinite(limit) or limit <= 0:
            limit = float(np.max(np.abs(data[i])) or 1.0)
        limits[i] = limit
    factor, unit = pick_unit(float(limits.max()))

    n_channels = len(channels)
    fig, axes = plt.subplots(
        n_channels, 1, figsize=(16, max(2.2, 2.0 * n_channels)), sharex=True
    )
    axes = np.atleast_1d(axes)

    for i, (ax, name) in enumerate(zip(axes, channels)):
        ax.plot(times, data[i] * factor, lw=0.6, color="#1f3a5f")
        for onset in event_onsets:
            ax.axvline(onset, color="tab:blue", ls="--", lw=0.8, alpha=0.8)
        if i == 0:
            # 事件文字只标在最上面那张图的顶部，避免四张图重复刷屏
            for onset in event_onsets:
                ax.text(
                    onset,
                    1.0,
                    TRIAL_DESCRIPTION,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="bottom",
                    ha="right",
                    fontsize=7,
                    color="tab:blue",
                )

        # 每条通道独立定纵轴（已按分位数截断，超出部分被裁掉）
        ax.set_ylim(-limits[i] * factor, limits[i] * factor)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_ylabel(f"{name} ({unit})", fontsize=8)
        ax.grid(axis="x", color="#dddddd", lw=0.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        clipped = np.mean(np.abs(data[i]) > limits[i]) * 100
        if clipped > 0.5:
            ax.text(
                0.995, 0.98, f"{clipped:.1f}% clipped", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#999999",
            )

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"{title}  |  {', '.join(channels)}")
    axes[-1].set_xlim(times[0], times[-1])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制 EDF 辅助通道（EKG / EMG / VEO / HEO）的波形。"
    )
    parser.add_argument("--subject", required=True, help="被试，如 01 或 sub-01")
    parser.add_argument("--ses", required=True, help="session，如 1、01 或 ses-01")
    parser.add_argument("--task", default="imagine", help="任务名，默认 imagine")
    parser.add_argument("--run", required=True, help="run 编号，如 1、01、010")
    parser.add_argument("--start", type=float, default=0.0, help="窗口起点（秒），默认 0")
    parser.add_argument("--end", type=float, default=None, help="窗口终点（秒），省略则到文件末尾")
    parser.add_argument(
        "--channels", nargs="+", default=None,
        help=f"只画指定通道，默认 {' / '.join(AUX_CHANNELS)}",
    )
    parser.add_argument("--out", type=Path, default=None, help="输出 PNG 路径（默认 figures/ 下自动命名）")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI，默认 150")
    args = parser.parse_args()

    subject = normalize_subject(args.subject)
    session = normalize_session(args.ses)
    run = normalize_run(args.run)

    edf_file = find_edf(subject, session, args.task, run)
    raw, sfreq, duration, trial_onsets = load_segment(edf_file, args.start, args.end)
    channels = resolve_channels(raw, args.channels)

    end_time = raw.times[-1] + args.start
    window_onsets = trial_onsets[
        (trial_onsets >= args.start) & (trial_onsets <= end_time)
    ]
    out_path = args.out or (
        DEFAULT_OUT_DIR
        / f"aux_waveform_{subject}_{session}_task-{args.task}_run-{run_label(run)}_"
        f"{args.start:.1f}-{end_time:.1f}s.png"
    )

    print(f"[File]   {edf_file}")
    print(f"[Info]   sfreq={sfreq:g} Hz, file duration={duration:.1f} s")
    print(f"[Window] {args.start:.3f} - {end_time:.3f} s ({end_time - args.start:.1f} s)")
    print(f"[Plot]   {len(channels)} channels: {channels}")
    print(f"[Events] {len(window_onsets)} trial onsets in window: "
          f"{np.round(window_onsets, 3).tolist()}")

    # 顺手打印幅度量级，方便判断哪个通道需要单独处理
    stats = raw.get_data(picks=channels) * 1e3
    for name, row in zip(channels, stats):
        print(f"[Amp]    {name}: ptp={np.ptp(row):.3g} mV, "
              f"range=[{row.min():.3g}, {row.max():.3g}] mV")

    plot_aux(raw, channels, out_path, args.dpi, edf_file.name, args.start, window_onsets)
    print(f"[Saved]  {out_path}")


if __name__ == "__main__":
    main()
