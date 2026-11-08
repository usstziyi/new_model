"""统计 preprocessed_pkl 中各被试的试次数与句子重复情况。

pkl 结构：``pickle.load`` 得到一个 list，每个元素是一个试次 dict::

    {"text": "附近还有其他的酒店吗", "input_features": ndarray(1, 125, 1651)}

一个试次对应一句话，因此「句数」等于试次数，区别在于是否有重复句子。

用法::

    # 统计全部被试
    uv run scripts/stats_pkl_sentences.py

    # 只统计指定被试
    uv run scripts/stats_pkl_sentences.py --subjects sub-01 sub-02
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL_ROOT = (
    REPO_ROOT / "datasets" / "ds005170-download" / "derivatives" / "preprocessed_pkl"
)

# 文件名形如 sub-01_task-imagine_run-01_eeg.pkl
PKL_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>\d+)_eeg\.pkl$"
)


def normalize_subject(subject: str) -> str:
    """把 01 / sub-01 统一成 sub-01。"""
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def collect_subject(subject_dir: Path) -> tuple[dict[str, Counter], int]:
    """读取一个被试的所有 pkl。

    返回 (task -> 该 task 内句子计数, pkl 文件数)。
    句子在返回前已 strip，空句子也保留（记为 ""），便于统计缺失文本。
    重复统计只在该被试内部进行，不跨 task、不跨被试比较。
    """
    sentences_per_task: dict[str, Counter] = {}
    n_files = 0

    for pkl_file in sorted(subject_dir.rglob("*.pkl")):
        match = PKL_PATTERN.match(pkl_file.name)
        if match is None:
            print(f"[Skip]   文件名不符合规则: {pkl_file}")
            continue

        with pkl_file.open("rb") as handle:
            items = pickle.load(handle)

        n_files += 1
        counter = sentences_per_task.setdefault(match["task"], Counter())
        for item in items:
            counter[str(item.get("text", "")).strip()] += 1

    return sentences_per_task, n_files


def duplicate_stats(sentences: Counter) -> tuple[int, int, int, int]:
    """返回 (句子总数, 唯一句子数, 重复句子种类数, 冗余重复次数)。

    冗余重复次数 = 句子总数 - 唯一句子数，即「多出来的、完全重复的句子」条数。
    """
    n_total = sum(sentences.values())
    n_unique = len(sentences)
    n_dup_kinds = sum(1 for count in sentences.values() if count > 1)
    return n_total, n_unique, n_dup_kinds, n_total - n_unique


def report(subject: str, sentences_per_task: dict[str, Counter], n_files: int, top: int) -> None:
    """打印单个被试的统计结果（重复只在该被试内部统计）。"""
    merged: Counter = Counter()
    for counter in sentences_per_task.values():
        merged.update(counter)

    n_trials = sum(merged.values())
    n_unique = len(merged)
    n_dup_kinds = sum(1 for count in merged.values() if count > 1)
    n_extra = n_trials - n_unique
    duplicates = {text: count for text, count in merged.items() if count > 1}

    print(f"=== {subject}（被试内部统计）===")
    print(f"  pkl 文件数        : {n_files}")
    print(f"  试次总数          : {n_trials} -> "
          + ", ".join(f"{task}={sum(c.values())}" for task, c in sorted(sentences_per_task.items())))
    print(f"  句子总数          : {n_trials}")
    print(f"  唯一句子数        : {n_unique}")
    print(f"  重复句子种类      : {n_dup_kinds} ({n_dup_kinds / n_unique:.1%} 的句子出现过不止一次)")
    print(f"  冗余重复次数      : {n_extra} ({n_extra / n_trials:.1%} 的试次是重复句子)")

    if merged.get(""):
        print(f"  空句子            : {merged['']}")

    print("  按 task 拆分（各自内部去重）:")
    for task, counter in sorted(sentences_per_task.items()):
        t_total, t_unique, t_dup_kinds, t_extra = duplicate_stats(counter)
        print(f"    {task:<8}: 句子={t_total}, 唯一={t_unique}, "
              f"重复种类={t_dup_kinds}, 冗余重复={t_extra}")

    if duplicates:
        print(f"  重复次数 Top {min(top, len(duplicates))}:")
        for text, count in Counter(duplicates).most_common(top):
            print(f"    {count:>3}x  {text!r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 preprocessed_pkl 中各被试的试次数、句子数与重复句子。"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="要统计的被试，如 01 02 或 sub-01 sub-02；默认统计目录下全部被试",
    )
    parser.add_argument(
        "--pkl-root", type=Path, default=DEFAULT_PKL_ROOT,
        help="preprocessed_pkl 根目录",
    )
    parser.add_argument("--top", type=int, default=10, help="每个被试展示的重复句子条数，默认 10")
    args = parser.parse_args()

    if not args.pkl_root.exists():
        raise FileNotFoundError(f"pkl 根目录不存在: {args.pkl_root}")

    if args.subjects is None:
        subjects = sorted(
            path.name for path in args.pkl_root.iterdir() if path.is_dir()
        )
    else:
        subjects = [normalize_subject(subject) for subject in args.subjects]

    print(f"[Root]   {args.pkl_root}\n")

    for subject in subjects:
        subject_dir = args.pkl_root / subject
        if not subject_dir.exists():
            print(f"[Warn]   被试目录不存在，跳过: {subject_dir}\n")
            continue
        sentences_per_task, n_files = collect_subject(subject_dir)
        report(subject, sentences_per_task, n_files, args.top)


if __name__ == "__main__":
    main()
