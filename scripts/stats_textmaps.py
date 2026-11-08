"""统计 textmaps.json 中的句子总数、种类数以及每个种类包含多少句。

``textmaps.json`` 是「句子 -> 类别编号」的映射，键本身唯一，因此一个键就是一句话；
``classnumber.json`` 是「类别编号 -> 类别名称」的映射，用来给类别编号贴名字。

用法::

    uv run scripts/stats_textmaps.py
    uv run scripts/stats_textmaps.py --sort count
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_ROOT = REPO_ROOT / "datasets" / "ds005170-download" / "json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 textmaps.json 的句子数、类别数与各类别句子数。"
    )
    parser.add_argument(
        "--json-root", type=Path, default=DEFAULT_JSON_ROOT,
        help="存放 textmaps.json / classnumber.json 的目录",
    )
    parser.add_argument(
        "--sort", choices=("index", "count", "name"), default="index",
        help="输出排序方式：按类别编号（默认）、按句子数降序、按名称",
    )
    args = parser.parse_args()

    textmaps = load_json(args.json_root / "textmaps.json")
    classnumber = load_json(args.json_root / "classnumber.json")

    # 句子 -> 类别编号；编号统一成 int，名称表里是字符串键
    counts = Counter(int(label) for label in textmaps.values())
    total = len(textmaps)

    print(f"[File]   {args.json_root / 'textmaps.json'}")
    print(f"句子总数 : {total}")
    print(f"类别总数 : {len(counts)}（名称表里共 {len(classnumber)} 类）")
    print()

    rows = [
        (index, classnumber.get(str(index), f"<未知类别 {index}>"), count)
        for index, count in counts.items()
    ]
    if args.sort == "count":
        rows.sort(key=lambda row: (-row[2], row[0]))
    elif args.sort == "name":
        rows.sort(key=lambda row: row[1])
    else:
        rows.sort(key=lambda row: row[0])

    width = max(len(name) for _, name, _ in rows)
    print(f"{'编号':>4}  {'类别名称':<{width}}  {'句子数':>6}  占比")
    for index, name, count in rows:
        print(f"{index:>4}  {name:<{width}}  {count:>6}  {count / total:.2%}")

    empty = [
        (index, classnumber[str(index)])
        for index in sorted(int(key) for key in classnumber)
        if index not in counts
    ]
    if empty:
        print("\n没有句子的类别:")
        for index, name in empty:
            print(f"{index:>4}  {name}")

    values = list(counts.values())
    print(f"\n每类句子数: 最小 {min(values)}, 最大 {max(values)}, 平均 {total / len(values):.1f}")


if __name__ == "__main__":
    main()
