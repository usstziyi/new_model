"""绘制某个 EDF 的 EEG 波形，用于快速肉眼检查原始信号。

参数（秒）：
    --subject   被试，支持 "01" / "sub-01"
    --ses       session，支持 "1" / "01" / "ses-01"
    --task      任务名，默认 imagine
    --run       run 编号，支持 "1" / "01" / "010"
    --start     窗口起点，默认 0
    --end       窗口终点，省略时画到文件末尾

用法::

    # 看 run-01 开头 20 秒
    uv run scripts/plot_edf_waveform.py --subject sub-01 --ses 01 --run 01 --start 0 --end 20

    # 看第一个试次附近（run-01 首个 65380 事件在 89.5 s）
    uv run scripts/plot_edf_waveform.py --subject 01 --ses ses-01 --run 010 --start 85 --end 105

    # 只看前若干个通道
    uv run scripts/plot_edf_waveform.py --subject sub-01 --ses 01 --run 01 --start 85 --end 105 --channels 10

    # 看整段（点数会自动抽稀）
    uv run scripts/plot_edf_waveform.py --subject sub-01 --ses 01 --run 045

默认画该文件全部 EEG 通道（Trigger 是 stim 通道，不在其中），``--channels N``
只画前 N 个；窗口内的试次起点（Trigger 事件 65380）用蓝色虚线竖线标出，并在
顶部竖排标注 ``trial_start``。图保存在
``figures/waveform_<subject>_<ses>_task-<task>_run-<run>_<start>-<end>s.png``。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只存图，不依赖图形界面

import matplotlib.pyplot as plt
import mne
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "ds005170-download"
DEFAULT_OUT_DIR = REPO_ROOT / "figures"

# 文件名形如 sub-01_ses-01_task-imagine_run-010_eeg.edf
EDF_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_(?P<session>ses-\d+)_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>\d+)_eeg\.edf$"
)

# 每通道最多画这么多点，超出就按步长抽稀（只影响显示，不改数据）
MAX_POINTS_PER_CHANNEL = 4000

# 这些通道在 EDF 里也被标为 eeg，但幅度比脑电大几个量级（EMG 峰峰值可达 46 V），
# 默认不画，否则会把整张图的纵轴撑爆；需要时用 --channels 显式指定
AUX_CHANNELS = ("VEO", "HEO", "EKG", "EMG")

# Trigger 通道与试次起点事件码（与 preprocessing.py 一致）
STIM_CHANNEL = "Trigger"
TRIAL_EVENT_CODE = 65380
TRIAL_DESCRIPTION = "trial_start"


def normalize_subject(subject: str) -> str:
    """统一 subject 格式："1" / "01" / "sub-01" -> "sub-01"。"""
    number = subject.strip().removeprefix("sub-")
    try:
        return f"sub-{int(number):02d}"
    except ValueError as e:
        raise ValueError(f"非法 subject: {subject!r}") from e


def normalize_session(session: str) -> str:
    """统一 session 格式："1" / "01" / "ses-01" -> "ses-01"。"""
    number = session.strip().removeprefix("ses-")
    try:
        return f"ses-{int(number):02d}"
    except ValueError as e:
        raise ValueError(f"非法 ses: {session!r}") from e


def normalize_run(run: str) -> int:
    """统一 run 格式："1" / "01" / "010" / "run-01" -> 1。"""
    number = run.strip().removeprefix("run-")
    try:
        return int(number)
    except ValueError as e:
        raise ValueError(f"非法 run: {run!r}") from e


def run_label(run: int) -> str:
    """按原始文件名习惯补零：run 1-9 -> "01"，run 10+ -> "010"。"""
    return f"{run:02d}" if run < 10 else f"{run:03d}"


def find_edf(subject: str, session: str, task: str, run: int) -> Path:
    """按 subject / session / task / run 定位唯一的 EDF（容忍文件名零填充差异）。"""
    eeg_dir = DATASET_ROOT / subject / session / "eeg"
    if not eeg_dir.exists():
        raise FileNotFoundError(f"EEG 目录不存在: {eeg_dir}")

    matches = []
    for edf in sorted(eeg_dir.glob("*.edf")):
        match = EDF_PATTERN.match(edf.name)
        if match is None:
            continue
        if (
            match["subject"] == subject
            and match["session"] == session
            and match["task"] == task
            and int(match["run"]) == run
        ):
            matches.append(edf)

    if not matches:
        raise FileNotFoundError(
            f"未找到 EDF: subject={subject}, ses={session}, task={task}, run={run} (目录 {eeg_dir})"
        )
    if len(matches) > 1:
        raise RuntimeError("匹配到多个 EDF:\n" + "\n".join(str(path) for path in matches))
    return matches[0]


def load_segment(edf_file: Path, start: float, end: float | None):
    """只读取 [start, end) 这一段，避免整段 EDF（约 1000 Hz x 20+ min）读入内存。

    返回 (raw, sfreq, duration, trial_onsets)：trial_onsets 是全文件里
    65380 事件的绝对时间（秒），用于在图上画红色竖线。
    """
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq

    if not 0 <= start < duration:
        raise ValueError(f"start={start} 超出文件范围 [0, {duration:.3f}]")
    tmax = duration if end is None else min(end, duration)
    if tmax <= start:
        raise ValueError(f"end={tmax} 必须大于 start={start}")

    # 事件在裁剪前从 Trigger 通道提取，换算成秒后与采样率、窗口无关
    events = mne.find_events(raw, stim_channel=STIM_CHANNEL, verbose=False)
    trial_onsets = events[events[:, 2] == TRIAL_EVENT_CODE, 0] / sfreq

    raw.crop(tmin=start, tmax=tmax)
    raw.load_data(verbose=False)
    return raw, sfreq, duration, trial_onsets


def plot_segment(
    raw,
    out_path: Path,
    dpi: int,
    title: str,
    window_start: float,
    event_onsets: np.ndarray,
) -> None:
    """按通道垂直堆叠画波形（单位 µV），通道数多时自动压缩画布字号。"""
    picks = raw.ch_names  # load_segment 后已按 --channels 选好

    data = raw.get_data(picks=picks) * 1e6  # V -> µV
    # crop 后 raw.times 从 0 开始，加回窗口起点还原绝对时间
    times = raw.times + window_start
    decim = max(1, int(np.ceil(data.shape[1] / MAX_POINTS_PER_CHANNEL)))
    data = data[:, ::decim]
    times = times[::decim]

    # 去直流后按 90 分位峰峰值定通道间距，避免个别大漂移把整张图撑开
    data = data - np.median(data, axis=1, keepdims=True)
    spacing = float(np.percentile(np.ptp(data, axis=1), 90)) * 1.2
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0

    n_channels = data.shape[0]
    offsets = (n_channels - 1 - np.arange(n_channels)) * spacing

    fig, ax = plt.subplots(figsize=(16, max(4.0, 0.16 * n_channels)))
    for i in range(n_channels):
        ax.plot(times, data[i] + offsets[i], lw=0.5, color="#1f3a5f")

    # 事件画法照 plot_fif_waveform.py：蓝色虚线竖线 + 顶部竖排 event 描述
    top = (n_channels - 1) * spacing + 0.8 * spacing
    text_size = 7 if n_channels <= 40 else 4
    for onset in event_onsets:
        ax.axvline(onset, color="tab:blue", ls="--", lw=0.8, alpha=0.8)
        ax.text(
            onset,
            top,
            TRIAL_DESCRIPTION,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=text_size,
            color="tab:blue",
        )

    font_size = 6 if n_channels <= 40 else 4.5 if n_channels <= 120 else 3.5
    ax.set_yticks(offsets)
    ax.set_yticklabels(picks, fontsize=font_size)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.8 * spacing, top + 2.2 * spacing)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{title}  |  {n_channels} channels")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制某个 EDF 的 EEG 波形（按 subject / ses / task / run + 时间窗定位）。"
    )
    parser.add_argument("--subject", required=True, help="被试，如 01 或 sub-01")
    parser.add_argument("--ses", required=True, help="session，如 1、01 或 ses-01")
    parser.add_argument("--task", default="imagine", help="任务名，默认 imagine")
    parser.add_argument("--run", required=True, help="run 编号，如 1、01、010")
    parser.add_argument("--start", type=float, default=0.0, help="窗口起点（秒），默认 0")
    parser.add_argument("--end", type=float, default=None, help="窗口终点（秒），省略则到文件末尾")
    parser.add_argument("--channels", type=int, default=None, help="只画前 N 个 EEG 通道，省略则画全部")
    parser.add_argument("--out", type=Path, default=None, help="输出 PNG 路径（默认 figures/ 下自动命名）")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI，默认 150")
    args = parser.parse_args()

    subject = normalize_subject(args.subject)
    session = normalize_session(args.ses)
    run = normalize_run(args.run)

    edf_file = find_edf(subject, session, args.task, run)
    raw, sfreq, duration, trial_onsets = load_segment(edf_file, args.start, args.end)

    if args.channels is not None and args.channels <= 0:
        parser.error("--channels 必须是正整数")

    raw.pick("eeg", verbose=False)
    aux = [ch for ch in raw.ch_names if ch in AUX_CHANNELS]
    if aux:
        raw.drop_channels(aux)
        print(f"[Skip]   aux channels not plotted: {aux}")

    if args.channels is not None:
        n_channels = min(args.channels, len(raw.ch_names))
        if n_channels < args.channels:
            print(f"[Warn]   只有 {len(raw.ch_names)} 个 EEG 通道，按 {n_channels} 个绘制")
        raw.pick(raw.ch_names[:n_channels], verbose=False)

    end_time = raw.times[-1] + args.start
    window_onsets = trial_onsets[
        (trial_onsets >= args.start) & (trial_onsets <= end_time)
    ]
    out_path = args.out or (
        DEFAULT_OUT_DIR
        / f"waveform_{subject}_{session}_task-{args.task}_run-{run_label(run)}_"
        f"{args.start:.1f}-{end_time:.1f}s.png"
    )

    print(f"[File]   {edf_file}")
    print(f"[Info]   sfreq={sfreq:g} Hz, file duration={duration:.1f} s")
    print(f"[Window] {args.start:.3f} - {end_time:.3f} s ({end_time - args.start:.1f} s)")
    print(f"[Plot]   {len(raw.ch_names)} channels")
    print(f"[Events] {len(window_onsets)} trial onsets in window: "
          f"{np.round(window_onsets, 3).tolist()}")

    plot_segment(raw, out_path, args.dpi, edf_file.name, args.start, window_onsets)
    print(f"[Saved]  {out_path}")


if __name__ == "__main__":
    main()
