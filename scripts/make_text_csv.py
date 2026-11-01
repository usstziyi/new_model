"""遍历 derivatives/preprocessed_pkl，按被试导出句表 CSV。

输出（每个被试卷两份）：
    <out-dir>/<subject>_reading.csv   对应 task-read（阅读阶段）
    <out-dir>/<subject>_imagine.csv   对应 task-imagine（想象阶段）

列：
    index        去重后从 0 递增的全局序号，顺序与 pkl 列表顺序一致
    day          实验天数，每 9 个 run 记作 1 天（run 1-9 -> day 1，10-18 -> day 2，…）
    run          run 编号（1 ~ 45）
    pos          该 run 内从 0 递增的位置，等价于 pkl 列表下标，用于唯一取回 EEG 数据
    sentence     句子原文
    n_chars      非标点字符个数（按 Unicode 类别 P* 剔除标点；空白也不计入）
    type_number  句子语义类别 id（0 ~ 38），查 json/textmaps.json 得到
    type_name    语义类别名称（中文），用 type_number 查 json/classnumber.json 得到

去重：
    遍历完某个「被试 x 任务」的全部 run 之后，以这张 CSV 为单位按 sentence 去重，
    同一句话只保留第一次出现的那条记录（对应最早 run / run 内最小 pos 的试次），
    其余同句试次直接丢弃；index 在去重后重新连续编号。

用法：
    python scripts/make_text_csv.py                          # 自动发现并处理全部被试
    python scripts/make_text_csv.py --subject sub-01 sub-03   # 只处理指定被试
    python scripts/make_text_csv.py --pkl-dir <目录> --out-dir <目录>
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "ds005170-download"
DEFAULT_PKL_DIR = DATASET_ROOT / "derivatives" / "preprocessed_pkl"
DEFAULT_TEXTMAPS = DATASET_ROOT / "json" / "textmaps.json"
DEFAULT_CLASSNUMBER = DATASET_ROOT / "json" / "classnumber.json"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "preprocessed"

# 文件名形如 sub-01_task-read_run-010_eeg.pkl
FILE_PATTERN = re.compile(r"^(?P<subject>sub-\d+)_task-(?P<task>[a-z]+)_run-(?P<run>\d+)_eeg\.pkl$")

# 任务名 -> 输出文件名后缀（保持用户指定的 reading / imagine 命名）
TASK_LABEL = {"read": "reading", "imagine": "imagine"}

# 每 9 个 run 划分为 1 天（该数据集每人每天 9 个 block，共 5 天 / 45 个 run）
RUNS_PER_DAY = 9
CSV_FIELDS = ["index", "day", "run", "pos", "sentence", "n_chars", "type_number", "type_name"]


def count_n_chars(sentence: str) -> int:
    """统计非标点字符个数：排除 Unicode 标点类（P*）与空白字符。"""
    return sum(
        1
        for ch in sentence
        if not ch.isspace() and unicodedata.category(ch)[0] != "P"
    )


def parse_name(file_name: str) -> tuple[str, str, int] | None:
    """解析文件名，返回 (subject, task, run)；不符合命名规范时返回 None。"""
    match = FILE_PATTERN.match(file_name)
    if match is None:
        return None
    return match["subject"], match["task"], int(match["run"])


def discover_subjects(pkl_dir: Path) -> list[str]:
    """扫描目录，返回排序后的被试列表。"""
    subjects = set()
    for path in pkl_dir.rglob("sub-*_task-*_eeg.pkl"):
        parsed = parse_name(path.name)
        if parsed is not None:
            subjects.add(parsed[0])
    return sorted(subjects)


def collect(pkl_dir: Path, subject: str) -> dict[str, list[Path]]:
    """收集某被试的全部 pkl，按任务分组；组内按 run 编号数值升序。

    文件名是 run-01 / run-010 这类非等宽命名，直接按字符串排序会把 run-010
    排到 run-01 前面，故按解析出的 run 编号排序。
    """
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in pkl_dir.rglob(f"{subject}_task-*_eeg.pkl"):
        parsed = parse_name(path.name)
        if parsed is None:
            continue
        _, task, run = parsed
        grouped.setdefault(task, []).append((run, path))
    return {task: [path for _, path in sorted(items)] for task, items in grouped.items()}


def write_csv(
    files: list[Path], out_path: Path, textmaps: dict[str, int], classnames: dict[int, str]
) -> tuple[list[dict], int, int]:
    """把某被试某个任务下所有 run 的句子写成一张 CSV。

    返回 (写入的行, 自检不一致行数, 去重丢弃的行数)。自检项：由全局 index 反推的
    run 内偏移，应恒等于生成侧的 pos 计数器；两者不等说明行序与 pkl 列表顺序脱钩。
    自检在去重前完成，去重后 index 会重新编号。
    """
    rows: list[dict] = []
    mismatch = 0
    index = 0
    for path in files:
        _, _, run = parse_name(path.name)
        day = (run - 1) // RUNS_PER_DAY + 1
        base = index  # 该 run 首行在全表中的 index，用于独立反推偏移做自检
        with open(path, "rb") as f:
            items = pickle.load(f)
        for pos, item in enumerate(items):
            if index - base != pos:
                mismatch += 1
            sentence = item["text"]
            type_number = textmaps.get(sentence, "")
            rows.append(
                {
                    "index": index,
                    "day": day,
                    "run": run,
                    "pos": pos,
                    "sentence": sentence,
                    "n_chars": count_n_chars(sentence),
                    "type_number": type_number,
                    "type_name": classnames.get(type_number, ""),
                }
            )
            index += 1

    # 遍历完本 CSV 的全部 run 之后按句子去重：保留第一次出现的记录。
    # 行序与 pkl 列表顺序一致，所以"第一次"就是最早的 run / 最小的 pos。
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        if row["sentence"] in seen:
            continue
        seen.add(row["sentence"])
        row["index"] = len(deduped)  # 去重后重新连续编号
        deduped.append(row)
    n_dropped = len(rows) - len(deduped)
    rows = deduped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, mismatch, n_dropped


def report(
    task: str, files: list[Path], out_path: Path, rows: list[dict], mismatch: int, n_dropped: int
) -> None:
    """打印单个「被试 x 任务」的导出结果。"""
    n_chars = [r["n_chars"] for r in rows]
    runs = sorted({r["run"] for r in rows})
    days = sorted({r["day"] for r in rows})
    types = sorted({r["type_number"] for r in rows if r["type_number"] != ""})
    missing = [r["sentence"] for r in rows if r["type_number"] == ""]
    no_name = [r for r in rows if r["type_name"] == ""]
    odd = [r for r in rows if " " in r["sentence"] or "\u3000" in r["sentence"]]

    print(f"[{task:8s}] {len(files):2d} 个 run -> {out_path}")
    print(
        f"           句子数 {len(rows)}（去重丢弃 {n_dropped} 条） | "
        f"run 范围 {runs[0]}~{runs[-1]} | "
        f"n_chars min/mean/max = {min(n_chars)}/{sum(n_chars) / len(n_chars):.1f}/{max(n_chars)}"
    )
    print(f"           day 覆盖 {days}（每天 9 个 run）")
    print(f"           pos 自检（index 反推偏移 == pos）: 一致 {len(rows) - mismatch}/{len(rows)}")
    if mismatch:
        print(f"           [警告] {mismatch} 行不一致，行序可能与 pkl 列表脱钩")
    print(f"           type_number: 命中 {len(rows) - len(missing)}/{len(rows)}，共 {len(types)} 类（{types[0]}~{types[-1]}）")
    if missing:
        print(f"           [警告] {len(missing)} 条句子在 textmaps.json 中查不到；示例 {missing[:3]}")
    if no_name:
        print(f"           [警告] {len(no_name)} 条 type_number 在 classnumber.json 中查不到；示例 {no_name[0]['type_number']}")
    if odd:
        print(f"           注意：{len(odd)} 条句子含空白字符，已计入剔除；示例 {odd[0]['sentence']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="按被试从 preprocessed_pkl 导出句表 CSV")
    parser.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR, help="preprocessed_pkl 目录")
    parser.add_argument("--textmaps", type=Path, default=DEFAULT_TEXTMAPS, help="textmaps.json 路径（句子 -> 类别 id）")
    parser.add_argument(
        "--classnumber", type=Path, default=DEFAULT_CLASSNUMBER, help="classnumber.json 路径（类别 id -> 类别名）"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="CSV 输出目录")
    parser.add_argument("--subject", nargs="+", default=None, help="指定被试，可多个；默认自动发现全部")
    args = parser.parse_args()

    with open(args.textmaps, encoding="utf-8") as f:
        textmaps = json.load(f)
    with open(args.classnumber, encoding="utf-8") as f:
        # classnumber.json 的 key 是字符串形式的 id，转成 int 便于直接用 type_number 查
        classnames = {int(k): v for k, v in json.load(f).items()}

    subjects = args.subject or discover_subjects(args.pkl_dir)
    if not subjects:
        print(f"[错误] 在 {args.pkl_dir} 下没找到任何 sub-*_task-*_eeg.pkl")
        return 1

    print(f"待处理被试 {len(subjects)} 个: {', '.join(subjects)}")
    total = 0
    for subject in subjects:
        by_task = collect(args.pkl_dir, subject)
        if not by_task:
            print(f"[警告] {subject} 没有可用的 pkl，跳过")
            continue
        print(f"\n===== {subject} =====")
        for task, files in sorted(by_task.items()):
            label = TASK_LABEL.get(task, task)
            out_path = args.out_dir / f"{subject}_{label}.csv"
            rows, mismatch, n_dropped = write_csv(files, out_path, textmaps, classnames)
            total += len(rows)
            report(task, files, out_path, rows, mismatch, n_dropped)

    print(f"\n合计导出 {total} 行，输出目录 {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
