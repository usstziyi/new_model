"""扫描所有原始 EDF，找出波形中存在"信号恒定不变"的坏道并输出 CSV。

判定规则：
    逐通道逐采样点比较相邻采样值是否完全相等；若存在连续相等时长
    >= FLAT_SECONDS（默认 5 秒）的片段，记为一次"恒定段"。
    当一条通道的恒定段个数 **超过 MIN_FLAT_COUNT（默认 10 次）** 时，
    判定该通道为坏道。

判定前会先从 EEG 中剔除 EKG / EMG / VEO / HEO 四个辅助通道，
它们不是脑电通道，不能参与坏道判定。

输出列：
    file, subject, session, task, run, duration_s, n_eeg_channels,
    n_bad, bad_channels（分号分隔）, bad_flats（name=flats(N次, longest xx.xs@xx.xs)，分号分隔）

用法::

    # 全部被试
    uv run scripts/scan_bad_channels.py

    # 指定被试
    uv run scripts/scan_bad_channels.py --subject sub-01 sub-02

    # 调整恒定段时长 / 最少恒定段个数 / 分块长度 / 输出路径
    uv run scripts/scan_bad_channels.py --flat-seconds 5 --min-flat-count 10 --chunk-seconds 60
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "ds005170-download"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "bad_channels" / "bad_channels.csv"

EDF_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_(?P<session>ses-\d+)_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>\d+)_eeg\.edf$"
)

# 非 EEG 通道，判定前剔除
AUX_CHANNELS = {"EKG", "EMG", "VEO", "HEO"}

# 判定阈值
FLAT_SECONDS = 5.0    # 连续恒定不变的最短时长（秒）
MIN_FLAT_COUNT = 10   # 恒定段个数超过它才判为坏道
CHUNK_SECONDS = 60.0  # 分块读取长度（秒），只为控制内存

CSV_FIELDS = [
    "file", "subject", "session", "task", "run", "duration_s",
    "n_eeg_channels", "n_bad", "bad_channels", "bad_flats",
]


def normalize_subject(subject: str) -> str:
    """'1' / '01' / 'sub-01' -> 'sub-01'。"""
    number = subject.strip().removeprefix("sub-")
    try:
        return f"sub-{int(number):02d}"
    except ValueError as e:
        raise ValueError(f"非法 subject: {subject!r}") from e


def iter_edf_files(subjects: list[str] | None):
    """按 subject/session/run 排序遍历 EDF。"""
    if subjects:
        files = []
        for subject in subjects:
            subject_dir = DATASET_ROOT / subject
            if not subject_dir.exists():
                raise FileNotFoundError(f"被试目录不存在: {subject_dir}")
            files.extend(sorted(subject_dir.rglob("*_eeg.edf")))
    else:
        files = sorted(DATASET_ROOT.rglob("sub-*/ses-*/eeg/*_eeg.edf"))
    return files


def relative_name(edf_file: Path) -> str:
    """相对数据集根目录的路径；传入相对路径时也能工作。"""
    try:
        return edf_file.resolve().relative_to(DATASET_ROOT).as_posix()
    except ValueError:
        return edf_file.name


def analyze_file(
    edf_file: Path,
    flat_seconds: float = FLAT_SECONDS,
    chunk_seconds: float = CHUNK_SECONDS,
    min_flat_count: int = MIN_FLAT_COUNT,
) -> dict:
    """扫描一个 EDF，返回恒定段个数超过阈值的通道。"""
    edf_file = edf_file.resolve()
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)

    # 先剔除辅助通道，再只保留 EEG
    aux = [ch for ch in raw.ch_names if ch in AUX_CHANNELS]
    if aux:
        raw.drop_channels(aux)
    raw.pick("eeg", verbose=False)

    names = list(raw.ch_names)
    if not names:
        raise RuntimeError(f"没有 EEG 通道: {edf_file}")

    sfreq = float(raw.info["sfreq"])
    n_times = raw.n_times
    duration = n_times / sfreq

    n_ch = len(names)
    flat_samples = max(1, int(round(flat_seconds * sfreq)))  # 连续相等的最少采样点数

    # 每条通道：已判定完成的恒定段个数、最长恒定段长度及起始时间
    n_flats = np.zeros(n_ch, dtype=np.int64)
    longest_len = np.zeros(n_ch, dtype=np.int64)   # 采样点数
    longest_start = np.zeros(n_ch, dtype=np.int64)  # 采样点索引（文件内绝对）

    # 跨块携带状态：每条通道"当前正在延伸的恒定段"已持续采样点数，
    # 以及该段的起始采样点索引（尚未结算）
    run_len = np.zeros(n_ch, dtype=np.int64)
    run_start = np.full(n_ch, -1, dtype=np.int64)

    chunk = max(1, int(round(chunk_seconds * sfreq)))
    prev_last = None  # 上一块每通道最后一个采样值，用于跨块判断相等

    for start in range(0, n_times, chunk):
        stop = min(start + chunk, n_times)
        data = raw.get_data(start=start, stop=stop)  # (n_ch, n)
        if data.shape[1] == 0:
            continue

        n = data.shape[1]

        # 构造"当前点与上一点是否相等"的布尔序列，长度 n
        # eq[:, j] 表示 data[:, start+j] 是否等于它的前一个采样点（文件内）
        eq = np.zeros((n_ch, n), dtype=bool)
        if prev_last is not None:
            eq[:, 0] = data[:, 0] == prev_last  # 块首与上一块末比较
            if n > 1:
                eq[:, 1:] = data[:, 1:] == data[:, :-1]
        else:
            # 文件第一块：第 0 个采样点没有前一个点，视为不相等
            if n > 1:
                eq[:, 1:] = data[:, 1:] == data[:, :-1]

        # 逐通道扫描 eq，累积/结算恒定段
        # 用向量化方式：找出每行 False 的位置作为段边界
        for ch in range(n_ch):
            row = eq[ch]
            # 找到所有 False 的位置（段结束于该位置之前）
            # 同时处理块首延续：run_len[ch] 已记录进入本块时正在延伸的长度
            # 为简洁与正确性，这里用状态机推进（每通道 O(n)）
            rl = run_len[ch]
            rs = run_start[ch]
            for j in range(n):
                if row[j]:
                    if rl == 0:
                        # 段起点：前一个采样点（文件内绝对索引 start+j-1）
                        rs = start + j - 1
                    rl += 1
                else:
                    if rl >= flat_samples:
                        n_flats[ch] += 1
                        if rl > longest_len[ch]:
                            longest_len[ch] = rl
                            longest_start[ch] = rs
                    rl = 0
                    rs = -1
            run_len[ch] = rl
            run_start[ch] = rs

        prev_last = data[:, -1].copy()

    # 文件结束，结算仍在延伸的恒定段
    for ch in range(n_ch):
        rl = run_len[ch]
        if rl >= flat_samples:
            n_flats[ch] += 1
            if rl > longest_len[ch]:
                longest_len[ch] = rl
                longest_start[ch] = run_start[ch]

    bad_channels, bad_flats = [], []
    for name, count, llen, lstart in zip(names, n_flats, longest_len, longest_start):
        if count <= min_flat_count:  # 次数不够，不算坏道
            continue
        bad_channels.append(name)
        longest_s = llen / sfreq
        start_s = lstart / sfreq
        bad_flats.append(
            f"{name}=flats({int(count)}次, longest {longest_s:.1f}s@{start_s:.1f}s)"
        )

    match = EDF_PATTERN.match(edf_file.name)
    meta = match.groupdict() if match else {}

    return {
        "file": relative_name(edf_file),
        "subject": meta.get("subject", ""),
        "session": meta.get("session", ""),
        "task": meta.get("task", ""),
        "run": meta.get("run", ""),
        "duration_s": round(float(duration), 1),
        "n_eeg_channels": len(names),
        "n_bad": len(bad_channels),
        "bad_channels": ";".join(bad_channels),
        "bad_flats": ";".join(bad_flats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 EDF 中信号恒定不变的坏道并输出 CSV。")
    parser.add_argument("--subject", nargs="+", default=None, help="只扫描指定被试，默认全部")
    parser.add_argument(
        "--flat-seconds", type=float, default=FLAT_SECONDS,
        help=f"连续恒定不变的最短时长（秒），默认 {FLAT_SECONDS:g}",
    )
    parser.add_argument(
        "--min-flat-count", type=int, default=MIN_FLAT_COUNT,
        help=f"恒定段个数超过该值才判为坏道，默认 {MIN_FLAT_COUNT}",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=CHUNK_SECONDS,
        help=f"分块读取长度（秒），默认 {CHUNK_SECONDS:g}",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 CSV 路径")
    args = parser.parse_args()

    subjects = [normalize_subject(s) for s in args.subject] if args.subject else None
    files = iter_edf_files(subjects)
    if not files:
        raise RuntimeError("没有找到 EDF 文件")

    print(
        f"[Scan] {len(files)} EDF files, "
        f"flat>={args.flat_seconds:g}s, min_flat_count>{args.min_flat_count}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()

        for index, edf_file in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {relative_name(edf_file)}")
            try:
                row = analyze_file(
                    edf_file, args.flat_seconds, args.chunk_seconds, args.min_flat_count
                )
            except Exception as exc:
                print(f"  [Error] {exc}")
                row = {
                    "file": relative_name(edf_file),
                    "subject": "", "session": "", "task": "", "run": "",
                    "duration_s": "", "n_eeg_channels": "", "n_bad": "",
                    "bad_channels": "", "bad_flats": f"error={exc}",
                }
            print(f"  bad ({row['n_bad']}): {row['bad_channels']}")
            writer.writerow(row)
            handle.flush()

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()