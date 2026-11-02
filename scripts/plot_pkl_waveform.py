"""绘制 preprocessed_pkl 里 EEG 的波形，用于检查预处理后的试次级信号。

pkl 结构：``pickle.load`` 得到一个 list，每个元素是一个试次 dict::

    {"text": "附近还有其他的酒店吗", "input_features": ndarray(1, 125, 1651)}

``input_features[0]`` 即 (通道数, 采样点数)，单位是 V。pkl 里没有保存通道名，
通道顺序与原始 EDF 一致，只是去掉了预处理里的 useless_channels
（``10/11/84/85/110/111/EKG/EMG``）：133 - 8 = 125，最后三个是 VEO / HEO / Trigger。
所以这里读同被试的 EDF 头，按同样顺序剔除后给纵轴贴通道名。

用法（run 编号是跨 session 的全局编号，pkl 文件名里没有 ses）::

    # 画 run-01 的第 0 个试次（全部通道）
    uv run scripts/plot_pkl_waveform.py --subject sub-01 --run 01 --trial 0

    # 画第 50 个试次，只看前 10 个通道
    uv run scripts/plot_pkl_waveform.py --subject 01 --run 45 --trial 50 --channels 0 1 2 3 4 5 6 7 8 9

    # 知道采样率时用秒作横轴
    uv run scripts/plot_pkl_waveform.py --subject sub-01 --run 01 --trial 0 --sfreq 250

图保存在 ``figures/pkl_waveform_<subject>_task-<task>_run-<run>_trial-<idx>.png``。
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只存图，不依赖图形界面

import matplotlib.pyplot as plt
import mne
import numpy as np

from plot_edf_waveform import (
    DEFAULT_OUT_DIR,
    normalize_run,
    normalize_subject,
    run_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "ds005170-download"
PKL_ROOT = DATASET_ROOT / "derivatives" / "preprocessed_pkl"

# 文件名形如 sub-01_task-imagine_run-01_eeg.pkl
PKL_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_eeg\.pkl$"
)

# 预处理阶段从 EDF 里剔掉的通道（与 preprocessing.py 的 useless_channels 一致）：
# pkl 的通道就是 EDF 通道顺序减掉这些，剩下的最后三个依次是 VEO / HEO / Trigger
USELESS_CHANNELS = ("10", "11", "84", "85", "110", "111", "EKG", "EMG")

# 每通道最多画这么多点，超出就按步长抽稀（只影响显示，不改数据）
MAX_POINTS_PER_CHANNEL = 4000

# 峰峰值超过「全体中位数」这么多倍的通道，判为辅助/异常通道，默认不画：
# 四个辅助通道（VEO/HEO/EKG/EMG）的幅度比脑电大几个量级，混进来会把图撑爆
OUTLIER_PTP_FACTOR = 100.0


def find_pkl(subject: str, task: str, run: int) -> Path:
    """按 subject / task / run 定位唯一的 pkl（容忍文件名零填充差异）。"""
    subject_dir = PKL_ROOT / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"被试目录不存在: {subject_dir}")

    matches = []
    for pkl in sorted(subject_dir.rglob("*.pkl")):
        match = PKL_PATTERN.match(pkl.name)
        if match is None:
            continue
        if (
            match["subject"] == subject
            and match["task"] == task
            and int(match["run"]) == run
        ):
            matches.append(pkl)

    if not matches:
        raise FileNotFoundError(
            f"未找到 pkl: subject={subject}, task={task}, run={run} (目录 {subject_dir})"
        )
    if len(matches) > 1:
        raise RuntimeError("匹配到多个 pkl:\n" + "\n".join(str(path) for path in matches))
    return matches[0]


def find_edf_header(subject: str, task: str, run: int) -> Path:
    """找一份 EDF 用来读通道顺序（只需头信息，不加载数据）。

    优先精确匹配 subject + task + run；read 任务在 EDF 里没有对应文件时，退回同
    run 任意任务，再退回该被试任意一个 EDF —— 通道顺序由采集硬件决定，任务间一致。
    该被试的 EDF 没下载时，最后退回其它被试的 EDF（同一套电极布局）。
    """
    patterns = [
        f"{subject}/ses-*/eeg/{subject}_*_task-{task}_run-{run_label(run)}_eeg.edf",
        f"{subject}/ses-*/eeg/*_task-*_run-{run_label(run)}_eeg.edf",
        f"{subject}/ses-*/eeg/*_eeg.edf",
        "sub-*/ses-*/eeg/*_eeg.edf",
    ]
    for pattern in patterns:
        matches = sorted(DATASET_ROOT.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"数据集里找不到任何 EDF，无法还原通道名: subject={subject}")


def load_channel_names(subject: str, task: str, run: int) -> list[str]:
    """按 EDF 通道顺序剔除 useless_channels，得到与 pkl 一致的通道名列表。"""
    edf_file = find_edf_header(subject, task, run)
    if not edf_file.name.startswith(f"{subject}_"):
        print(f"[Names]  {subject} 没有 EDF，改用 {edf_file.name} 的通道顺序")
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
    return [ch for ch in raw.ch_names if ch not in USELESS_CHANNELS]


def load_trial(pkl_file: Path, trial: int) -> tuple[np.ndarray, str, int]:
    """读取整个 pkl，返回 (该试次的 (通道, 采样点) 数组, 文本, 总试次数)。

    一个 pkl 文件对应一个 run，里面 100+ 个试次都序列化在一起，
    无法只反序列化单个试次，所以这里一次性加载。
    """
    with pkl_file.open("rb") as handle:
        items = pickle.load(handle)
    if not items:
        raise RuntimeError(f"pkl 为空: {pkl_file}")
    if not -len(items) <= trial < len(items):
        raise IndexError(
            f"{pkl_file.name} 只有 {len(items)} 个试次，trial={trial} 超出范围 "
            f"[-{len(items)}, {len(items) - 1}]"
        )

    item = items[trial]
    features = np.asarray(item["input_features"])
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]  # 去掉批次维
    if features.ndim != 2:
        raise RuntimeError(f"无法识别的 input_features 形状: {features.shape}")

    return features, str(item.get("text", "")), len(items)


def plot_trial(
    features: np.ndarray,
    channel_ids: np.ndarray,
    channel_names: list[str],
    out_path: Path,
    dpi: int,
    title: str,
    sfreq: float | None,
) -> None:
    """按通道垂直堆叠画波形（单位 µV），通道数多时自动压缩画布字号。"""
    data = features[channel_ids].astype(float) * 1e6  # V -> µV

    # 去掉每通道直流，避免某些通道的大偏置把整张图撑开
    data = data - np.median(data, axis=1, keepdims=True)

    n_samples = data.shape[1]
    times = np.arange(n_samples, dtype=float)
    xlabel = "Sample"
    if sfreq:
        times /= sfreq
        xlabel = "Time (s)"

    decim = max(1, int(np.ceil(n_samples / MAX_POINTS_PER_CHANNEL)))
    data = data[:, ::decim]
    times = times[::decim]

    # 按 90 分位峰峰值定通道间距，避免个别大漂移把间距拉爆
    spacing = float(np.percentile(np.ptp(data, axis=1), 90)) * 1.2
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0

    n_channels = data.shape[0]
    offsets = (n_channels - 1 - np.arange(n_channels)) * spacing

    fig, ax = plt.subplots(figsize=(16, max(4.0, 0.16 * n_channels)))
    for i in range(n_channels):
        ax.plot(times, data[i] + offsets[i], lw=0.6, color="#1f3a5f")

    font_size = 6 if n_channels <= 40 else 4.5 if n_channels <= 120 else 3.5
    ax.set_yticks(offsets)
    ax.set_yticklabels([channel_names[idx] for idx in channel_ids], fontsize=font_size)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.8 * spacing, (n_channels - 1) * spacing + 1.8 * spacing)
    ax.set_xlabel(xlabel)
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
        description="绘制 preprocessed_pkl 中某个试次的 EEG 波形（按 subject / task / run / trial 定位）。"
    )
    parser.add_argument("--subject", required=True, help="被试，如 01 或 sub-01")
    parser.add_argument("--task", default="imagine", help="任务名，默认 imagine（另有 read）")
    parser.add_argument("--run", required=True, help="run 编号，如 1、01、010（跨 session 的全局编号）")
    parser.add_argument("--trial", type=int, default=0, help="试次下标，默认 0（可用负数从末尾数）")
    parser.add_argument(
        "--channels", type=int, nargs="+", default=None,
        help="只画指定通道（0 起始下标），默认全部",
    )
    parser.add_argument(
        "--sfreq", type=float, default=None, help="采样率 Hz；提供时横轴用秒，否则用采样点序号",
    )
    parser.add_argument("--out", type=Path, default=None, help="输出 PNG 路径（默认 figures/ 下自动命名）")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI，默认 150")
    args = parser.parse_args()

    subject = normalize_subject(args.subject)
    run = normalize_run(args.run)

    pkl_file = find_pkl(subject, args.task, run)
    try:
        features, text, n_trials = load_trial(pkl_file, args.trial)
    except IndexError as exc:
        # 试次越界属于参数错误，转成 argparse 的报错，不打 traceback
        parser.error(str(exc))
    trial_index = args.trial if args.trial >= 0 else n_trials + args.trial

    n_channels_total, n_samples = features.shape

    # pkl 不存通道名，按 EDF 顺序剔除 useless_channels 还原
    channel_names = load_channel_names(subject, args.task, run)
    if len(channel_names) != n_channels_total:
        print(
            f"[Warn]   通道名数量 {len(channel_names)} 与 pkl 通道数 {n_channels_total} 不一致，"
            "纵轴退回序号标注"
        )
        channel_names = [f"ch{idx:03d}" for idx in range(n_channels_total)]

    # 完全恒定的通道（例如某些文件里被填成常数 65280）与幅度大几个量级的辅助
    # 通道，画出来要么只是占位、要么会把整张图撑爆，默认跳过
    ptp_uv = np.ptp(features, axis=1) * 1e6
    constant = [idx for idx in range(n_channels_total) if ptp_uv[idx] == 0]
    median_ptp = float(np.median(ptp_uv[ptp_uv > 0])) if np.any(ptp_uv > 0) else 0.0
    outliers = [
        idx for idx in range(n_channels_total)
        if ptp_uv[idx] > OUTLIER_PTP_FACTOR * median_ptp
    ]
    if constant:
        print(f"[Skip]   constant channels: {[channel_names[i] for i in constant]}")
    if outliers:
        print(
            f"[Skip]   outlier channels (ptp > {OUTLIER_PTP_FACTOR:g}x median): "
            f"{[channel_names[i] for i in outliers]}"
        )

    if args.channels is None:
        channel_ids = np.array(
            [
                idx for idx in range(n_channels_total)
                if idx not in constant and idx not in outliers
            ],
            dtype=int,
        )
    else:
        bad = [idx for idx in args.channels if not 0 <= idx < n_channels_total]
        if bad:
            parser.error(f"--channels 下标超出范围 [0, {n_channels_total - 1}]: {bad}")
        channel_ids = np.array(args.channels, dtype=int)

    if len(channel_ids) == 0:
        raise RuntimeError("没有可绘制的通道")

    out_path = args.out or (
        DEFAULT_OUT_DIR
        / f"pkl_waveform_{subject}_task-{args.task}_run-{run_label(run)}_trial-{trial_index}.png"
    )

    print(f"[File]   {pkl_file}")
    print(f"[Info]   trials={n_trials}, shape=(ch={n_channels_total}, samples={n_samples})")
    print(f"[Trial]  index={trial_index}, text={text!r}")
    print(f"[Plot]   {len(channel_ids)} channels: "
          f"{[channel_names[idx] for idx in channel_ids]}")
    print(f"[Amp]    ptp median={np.median(np.ptp(features[channel_ids], axis=1)) * 1e6:.1f} µV")

    plot_trial(
        features, channel_ids, channel_names, out_path, args.dpi, pkl_file.name, args.sfreq
    )
    print(f"[Saved]  {out_path}")


if __name__ == "__main__":
    main()
