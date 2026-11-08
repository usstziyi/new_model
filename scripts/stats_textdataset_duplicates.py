"""统计 textdataset 中 split_data_*.xlsx 的重复句子。

每个 xlsx 只有一张表，两列：``句子`` / ``标签``，每行一句话。
重复判定按句子文本（strip 后）完全一致；同时检查同一句子是否对应多个标签。

用法::

    uv run scripts/stats_textdataset_duplicates.py
    uv run scripts/stats_textdataset_duplicates.py --top 20
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_ROOT = REPO_ROOT / "datasets" / "ds005170-download" / "textdataset"

FILE_PATTERN = re.compile(r"^split_data_(?P<index>\d+)\.xlsx$")

SENTENCE_COLUMN = "句子"
LABEL_COLUMN = "标签"


def load_rows(text_root: Path) -> tuple[list[str], list[str], list[str], int]:
    """读取全部 xlsx。

    返回 (句子列表, 标签列表, 句子所属文件列表, 空句子数)。
    """
    files = [
        path for path in sorted(text_root.glob("*.xlsx"))
        if FILE_PATTERN.match(path.name)
    ]
    if not files:
        raise FileNotFoundError(f"目录下没有 split_data_*.xlsx: {text_root}")

    sentences: list[str] = []
    labels: list[str] = []
    sources: list[str] = []
    n_empty = 0

    for path in files:
        frame = pd.read_excel(path, sheet_name=0)
        missing = {SENTENCE_COLUMN, LABEL_COLUMN} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} 缺少列 {sorted(missing)}，实际列: {list(frame.columns)}")

        for text, label in zip(frame[SENTENCE_COLUMN], frame[LABEL_COLUMN]):
            text = "" if pd.isna(text) else str(text).strip()
            label = "" if pd.isna(label) else str(label).strip()
            if not text:
                n_empty += 1
                continue
            sentences.append(text)
            labels.append(label)
            sources.append(path.name)

    print(f"[Files]  {len(files)} 个 xlsx，跳过空句子 {n_empty} 行\n")
    return sentences, labels, sources, n_empty


def report(sentences: list[str], labels: list[str], sources: list[str], top: int) -> None:
    """打印整体重复统计。"""
    counts = Counter(sentences)
    n_total = len(sentences)
    n_unique = len(counts)
    n_dup_kinds = sum(1 for count in counts.values() if count > 1)
    n_extra = n_total - n_unique

    print("=== 整体统计 ===")
    print(f"  句子总数          : {n_total}")
    print(f"  唯一句子数        : {n_unique}")
    print(f"  重复句子种类      : {n_dup_kinds} ({n_dup_kinds / n_unique:.1%} 的句子出现过不止一次)")
    print(f"  冗余重复次数      : {n_extra} ({n_extra / n_total:.1%} 的句子是重复的)")

    # 同一句子对应多个标签：可能是标注不一致，会直接影响分类训练
    labels_of: dict[str, set[str]] = defaultdict(set)
    files_of: dict[str, set[str]] = defaultdict(set)
    for text, label, source in zip(sentences, labels, sources):
        labels_of[text].add(label)
        files_of[text].add(source)
    multi_label = {text: set_ for text, set_ in labels_of.items() if len(set_) > 1}
    cross_file = [text for text, files in files_of.items() if len(files) > 1]

    print(f"  同句多个标签      : {len(multi_label)}")
    print(f"  跨文件重复的句子  : {len(cross_file)}")

    # 出现次数的分布，便于判断重复规模
    dist = Counter(counts.values())
    print("  出现次数分布      : "
          + ", ".join(f"{times}次={kinds}句" for times, kinds in sorted(dist.items(), reverse=True)))

    # 每个类别的句子数（按行计数，含重复句子），从多到少
    label_counts = Counter(labels)
    name_width = max(len(label) for label in label_counts)
    print("\n=== 每个类别的句子数（从多到少）===")
    print(f"  {'类别':<{name_width}}  {'句子数':>6}  占比")
    for label, count in label_counts.most_common():
        print(f"  {label:<{name_width}}  {count:>6}  {count / n_total:.2%}")
    print(f"  类别总数: {len(label_counts)}")

    if multi_label:
        print(f"\n=== 同句多标签示例（最多 {top} 条）===")
        for text in sorted(multi_label, key=lambda t: -counts[t])[:top]:
            print(f"  {counts[text]:>3}x  {text!r} -> {sorted(multi_label[text])}")

    duplicates = {text: count for text, count in counts.items() if count > 1}
    if duplicates:
        print(f"\n=== 重复次数 Top {min(top, len(duplicates))} ===")
        for text, count in Counter(duplicates).most_common(top):
            print(f"  {count:>3}x  {text!r}  [{', '.join(sorted(labels_of[text]))}]  "
                  f"({', '.join(sorted(files_of[text]))})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 textdataset/split_data_*.xlsx 中的重复句子。"
    )
    parser.add_argument(
        "--text-root", type=Path, default=DEFAULT_TEXT_ROOT,
        help="textdataset 目录",
    )
    parser.add_argument("--top", type=int, default=10, help="示例展示条数，默认 10")
    args = parser.parse_args()

    if not args.text_root.exists():
        raise FileNotFoundError(f"目录不存在: {args.text_root}")

    sentences, labels, sources, _ = load_rows(args.text_root)
    report(sentences, labels, sources, args.top)


if __name__ == "__main__":
    main()
