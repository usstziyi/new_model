"""按类别导出 EEG 数据为独立 pkl。

从 ``<subject>_<task>.csv`` 统计 ``type_number`` 的样本数，取样本最多的前
``--top-k`` 类，再从 ``preprocessed_pkl/<subject>/eeg`` 中按 ``run`` + ``pos``
取出对应试次，把 122 个 EEG 通道（原始伏特，不做单位换算）按类别分别保存。

用法::

    uv run python scripts/export_top_classes.py
    uv run python scripts/export_top_classes.py --task imagine --subject sub-01 --top-k 10

每个输出文件形如 ``class_10.pkl``，内容是::

    {
        "subject": "sub-01",
        "task": "imagine",
        "type_number": 10,
        "type_name": "求职和职业发展",
        "sfreq": 500.0,
        "n_channels": 122,
        "unit": "V",
        "eeg": np.ndarray,        # (N, 122, T)，单位 V
        "text": list[str],        # 每个试次的句子
        "meta": list[dict],       # index / day / run / pos / n_chars
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed"
PKL_ROOT = PROJECT_ROOT / "datasets/ds005170-download/derivatives/preprocessed_pkl"
CLASSNUMBER = PROJECT_ROOT / "datasets/ds005170-download/json/classnumber.json"

SFREQ = 500
N_EEG_CHANNELS = 122
LABEL_COLUMN = "type_number"

TASK_ALIASES = {
    "reading": ("reading", "read"),
    "imagine": ("imagine", "imagine"),
}

_FILE_PATTERN = re.compile(r"_task-(?P<task>[a-z]+)_run-(?P<run>\d+)_eeg\.pkl$")


def parse_args():
    parser = argparse.ArgumentParser(description="按类别导出 EEG 数据为独立 pkl。")
    parser.add_argument("--subject", default="sub-01", help="被试编号，如 sub-01")
    parser.add_argument("--task", default="imagine", choices=("reading", "imagine"), help="任务")
    parser.add_argument("--top-k", type=int, default=10, help="导出样本数最多的前 k 类")
    parser.add_argument("--runs", nargs="+", type=int, default=None, help="只导出这些 run（默认全部）")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认 datasets/top<k>_<task>/<subject>")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="eeg 数组的 dtype")
    return parser.parse_args()


def load_class_names(path: Path = CLASSNUMBER) -> dict[int, str]:
    """读取 classnumber.json，返回 {type_number: 类别名}。"""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(key): name for key, name in raw.items()}


def read_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index_pkls(eeg_dir: Path, subject: str, pkl_task: str) -> dict[int, Path]:
    """按 run 编号索引 pkl 文件（文件名里的 run 号不等宽，不能按字符串排序）。"""
    paths: dict[int, Path] = {}
    for path in eeg_dir.glob(f"{subject}_task-{pkl_task}_run-*_eeg.pkl"):
        match = _FILE_PATTERN.search(path.name)
        if match is not None:
            paths[int(match["run"])] = path
    if not paths:
        raise FileNotFoundError(f"{eeg_dir} 下没有找到 {subject} 的 {pkl_task} pkl")
    return paths


def extract_eeg(features) -> np.ndarray:
    """``(1, 125, T)`` -> ``(122, T)`` float64，保持原始伏特值。"""
    array = np.asarray(features)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < N_EEG_CHANNELS:
        raise RuntimeError(f"无法识别的 input_features 形状: {array.shape}")
    return array[:N_EEG_CHANNELS].astype(np.float64)


def main():
    args = parse_args()
    label, pkl_task = TASK_ALIASES[args.task]
    csv_path = CSV_ROOT / f"{args.subject}_{label}.csv"
    rows = read_rows(csv_path)

    counts = Counter(int(row[LABEL_COLUMN]) for row in rows)
    top = counts.most_common(args.top_k)
    wanted = {class_id for class_id, _ in top}
    class_names = load_class_names()

    print(f"[CSV] {csv_path} 共 {len(rows)} 条，选出 {len(top)} 类：")
    for rank, (class_id, count) in enumerate(top, 1):
        print(f"  {rank:>2}. type_number={class_id:<2} {class_names.get(class_id, '?'):<12s} {count} 条")

    if args.runs is not None:
        keep = set(args.runs)
        rows = [row for row in rows if int(row["run"]) in keep]
        print(f"[Runs] 只保留 run={sorted(keep)}，剩余 {len(rows)} 条")

    rows = [row for row in rows if int(row[LABEL_COLUMN]) in wanted]
    by_run: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[int(row["run"])].append(row)

    eeg_dir = PKL_ROOT / args.subject / "eeg"
    pkl_paths = index_pkls(eeg_dir, args.subject, pkl_task)

    out_dir = args.out_dir or (PROJECT_ROOT / "datasets" / f"top{args.top_k}_{args.task}" / args.subject)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: dict[int, list[np.ndarray]] = {class_id: [] for class_id in wanted}
    texts: dict[int, list[str]] = {class_id: [] for class_id in wanted}
    metas: dict[int, list[dict]] = {class_id: [] for class_id in wanted}

    for run in sorted(by_run):
        if run not in pkl_paths:
            raise FileNotFoundError(f"缺少 run-{run} 的 pkl（目录 {eeg_dir}）")
        with pkl_paths[run].open("rb") as handle:
            items = pickle.load(handle)

        for row in by_run[run]:
            pos = int(row["pos"])
            if pos >= len(items):
                raise IndexError(f"run-{run} 的 pkl 只有 {len(items)} 条，pos={pos} 越界")
            class_id = int(row[LABEL_COLUMN])
            eeg = extract_eeg(items[pos]["input_features"])
            samples[class_id].append(eeg.astype(args.dtype))
            texts[class_id].append(row.get("sentence", ""))
            metas[class_id].append({
                "index": int(row["index"]),
                "day": int(row["day"]),
                "run": run,
                "pos": pos,
                "n_chars": int(row["n_chars"]),
            })
        print(f"[Run {run:>2}] 已提取 {len(by_run[run])} 条（累计 "
              f"{sum(len(v) for v in samples.values())} 条）")

    total_bytes = 0
    for class_id, count in top:
        arrays = samples[class_id]
        if not arrays:
            print(f"[Skip] type_number={class_id} 没有样本")
            continue
        lengths = {array.shape[-1] for array in arrays}
        if len(lengths) != 1:
            raise RuntimeError(f"type_number={class_id} 的试次长度不一致: {sorted(lengths)}")
        eeg = np.stack(arrays)
        payload = {
            "subject": args.subject,
            "task": args.task,
            "type_number": class_id,
            "type_name": class_names.get(class_id, ""),
            "sfreq": float(SFREQ),
            "n_channels": N_EEG_CHANNELS,
            "unit": "V",
            "eeg": eeg,
            "text": texts[class_id],
            "meta": metas[class_id],
        }
        out_path = out_dir / f"class_{class_id:02d}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        total_bytes += out_path.stat().st_size
        print(f"[Save] {out_path.name}  shape={eeg.shape}  {out_path.stat().st_size / 1e6:.1f} MB")
        samples[class_id] = []  # 尽早释放内存

    print(f"[Done] {len(top)} 个文件写入 {out_dir}，合计 {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
