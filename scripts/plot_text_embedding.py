"""把句表 CSV 里的句子送进文本编码器（LaBSE），画表征热力图和 text-text 余弦相似度热力图。

流程：
    <subject>_<task>.csv 的 sentence 列
        ↓  按 --types 选类别并合并成块（可选）
    LaBSETextEncoder（冻结，见 baseline/text_encoder.py）
        ↓
    (N, 768)，已 L2 归一化
        ↓                        ↘
    表征热力图                  余弦相似度 N x N 热力图

两张图分开保存，每个格子都是正方形（``square=True``），画布尺寸按格子数推导：
    figures/<subject>_<task>_text_embedding.png     N 行 x 768 列
    figures/<subject>_<task>_text_similarity.png    N 行 x N 列

格子边长按「每格多少英寸」控制，两张图各有一个参数（embedding 有 768 列，同样的
边长会让它比相似度图宽几十倍，所以默认值不同）：
    --embedding-cell-size    默认 0.05 英寸，画布宽约 768 x 0.05 = 38 英寸
    --similarity-cell-size   默认自适应，使整图约 10 英寸见方

用法：
    python scripts/plot_text_embedding.py --subject sub-01 --task imagine --types 13
    python scripts/plot_text_embedding.py --subject sub-01 --task imagine --types 13 19 26
    python scripts/plot_text_embedding.py --subject sub-01 --task reading --types 0 1 2 --skip-similarity
    python scripts/plot_text_embedding.py --csv datasets/preprocessed/sub-01_imagine.csv --types 25

``--types`` 按 ``type_number`` 选句子，并按参数里给出的顺序把同类句子合并成块（块内
保持 CSV 原顺序），相似度图上会用白色分割线标出类别块边界。不写 ``--types`` 就是
CSV 里的全部句子、保持原始顺序；全部句子有六千多条，N x N 的相似度图渲染不动，
这种情况要加 ``--skip-similarity`` 只画表征图。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只存图，不依赖图形界面

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.text_encoder import LaBSETextEncoder  # noqa: E402

DEFAULT_CSV_DIR = REPO_ROOT / "datasets" / "preprocessed"
DEFAULT_OUT_DIR = REPO_ROOT / "figures"

# 句子标签太长会把 y 轴挤爆，超过这个长度就截断
LABEL_MAX_CHARS = 12

# 画布在数据区域之外额外预留的空间（英寸），用于坐标标签和 colorbar
MARGIN_WIDTH = 3.0
MARGIN_HEIGHT = 1.6

# subplots 布局下 axes 只占 figure 的一部分，按这个比例反推画布尺寸，
# 否则 square=True 会让格子被可用空间压小
AXES_FRACTION = 0.78

# 相似度图没指定格子边长时，按这个目标尺寸反推（英寸）
SIMILARITY_TARGET_INCHES = 10.0


def setup_chinese_font() -> None:
    """优先使用系统中文字体，否则句子标签会显示成方框。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[警告] 系统中没有可用的中文字体，句子标签可能显示为方框")
    plt.rcParams["axes.unicode_minus"] = False


def resolve_csv(args: argparse.Namespace) -> Path:
    """按 --csv 或 --subject + --task 定位句表 CSV。"""
    if args.csv is not None:
        return args.csv
    if not args.subject or not args.task:
        raise SystemExit("[错误] 需要同时提供 --subject 和 --task，或者直接用 --csv 指定文件")
    return args.csv_dir / f"{args.subject}_{args.task}.csv"


def load_sentences(
    path: Path,
    types: list[int] | None,
) -> tuple[list[str], list[int], list[str]]:
    """读句表 CSV，返回 (句子列表, 每句的 type_number, 每句的 type_name)。

    ``types`` 非空时只保留这些类别的句子，并按参数中给出的顺序把同类句子合并成
    块（块内保持 CSV 原顺序）；不指定时返回全部句子、保持 CSV 原顺序。
    """
    if not path.is_file():
        raise SystemExit(f"[错误] CSV 不存在: {path}")
    # make_text_csv.py 写出的 CSV 带 BOM，用 utf-8-sig 兼容
    frame = pd.read_csv(
        path, encoding="utf-8-sig", usecols=["sentence", "type_number", "type_name"]
    )
    # type_number 一律转成整数：CSV 若被读成字符串，排序会按字典序走
    # （0, 1, 10, 11, ..., 2, 20...），这里先数值化再丢掉无效行
    frame["type_number"] = pd.to_numeric(frame["type_number"], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise SystemExit(f"[错误] {path} 里没有有效句子")
    frame["type_number"] = frame["type_number"].astype(int)
    if types is not None:
        frame = frame[frame["type_number"].isin(types)]
        if frame.empty:
            raise SystemExit(f"[错误] {path} 里没有 {'/'.join(map(str, types))} 这些类别的句子")
        # 按 --types 里写的顺序把类别块排出来，块内保持 CSV 原顺序
        order = {type_id: pos for pos, type_id in enumerate(types)}
        frame = frame.assign(_type_order=frame["type_number"].map(order))
        frame = frame.sort_values("_type_order", kind="stable").drop(columns="_type_order")
    return (
        frame["sentence"].astype(str).tolist(),
        frame["type_number"].tolist(),
        frame["type_name"].astype(str).tolist(),
    )


def report_types(type_ids: list[int], type_names: list[str]) -> None:
    """按类别在热力图中首次出现的顺序打印「类别: 个数」。

    顺序直接取自 ``type_ids``（它就是热力图的行顺序）：指定 ``--types`` 时等于
    参数里的类别顺序，不指定时等于 CSV 中的出现顺序。
    """
    counts = Counter(zip(type_ids, type_names))
    print(f"类别分布（{len(type_ids)} 句，{len(counts)} 类，顺序同热力图）：")
    printed: set[tuple[int, str]] = set()
    for key in zip(type_ids, type_names):
        if key in printed:
            continue
        printed.add(key)
        print(f"  {key[0]:02d} {key[1]}: {counts[key]}")


def block_boundaries(type_ids: list[int]) -> list[int]:
    """返回相邻两个类别块之间的分割位置（行索引，不含首尾边界）。"""
    return [i for i in range(1, len(type_ids)) if type_ids[i] != type_ids[i - 1]]


def short_label(sentence: str) -> str:
    return sentence if len(sentence) <= LABEL_MAX_CHARS else sentence[:LABEL_MAX_CHARS] + "…"


def square_figsize(rows: int, cols: int, cell_size: float) -> tuple[float, float]:
    """按「格子数 x 每格边长」算出画布尺寸，使每个格子是正方形。"""
    data_w = cols * cell_size / AXES_FRACTION
    data_h = rows * cell_size / AXES_FRACTION
    return data_w + MARGIN_WIDTH, data_h + MARGIN_HEIGHT


def save_figure(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(
    embeddings: np.ndarray,
    sentences: list[str],
    out_path: Path,
    cell_size: float,
    dpi: int,
) -> None:
    """画句子表征热力图 (N, 768)，每格正方形。"""
    n, dim = embeddings.shape
    labels = [short_label(s) for s in sentences]
    # embedding 有正有负，用 0 居中的发散色，按绝对值对称定色阶
    vmax = float(np.abs(embeddings).max())

    fig, ax = plt.subplots(figsize=square_figsize(n, dim, cell_size))
    sns.heatmap(
        embeddings,
        ax=ax,
        cmap="vlag",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        xticklabels=False,
        yticklabels=labels,
        cbar_kws={"label": "embedding 分量", "shrink": 0.5},
    )
    ax.set_xlabel(f"embedding 维度（{dim}）")
    ax.set_ylabel("句子")
    ax.set_title(f"LaBSE 句子表征（已 L2 归一化，{n} 句）")
    ax.tick_params(axis="y", labelsize=label_size(cell_size))

    save_figure(fig, out_path, dpi)


def plot_similarity(
    embeddings: np.ndarray,
    sentences: list[str],
    out_path: Path,
    cell_size: float,
    dpi: int,
    boundaries: list[int] | None = None,
) -> np.ndarray:
    """画 text-text 余弦相似度热力图 (N, N)，每格正方形；返回相似度矩阵。

    ``boundaries`` 给出类别块的分割位置时，在对应行列之间画白色分割线。
    """
    n = len(sentences)
    similarity = embeddings @ embeddings.T
    labels = [short_label(s) for s in sentences]

    fig, ax = plt.subplots(figsize=square_figsize(n, n, cell_size))
    sns.heatmap(
        similarity,
        ax=ax,
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        annot=n <= 20,
        fmt=".2f",
        annot_kws={"size": label_size(cell_size) * 0.8},
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "余弦相似度", "shrink": 0.5},
    )
    ax.set_xlabel("句子")
    ax.set_ylabel("句子")
    ax.set_title(f"text-text 余弦相似度（{n} 句）")
    ax.tick_params(axis="x", labelrotation=90, labelsize=label_size(cell_size))
    ax.tick_params(axis="y", labelrotation=0, labelsize=label_size(cell_size))

    # 第 k 行/列的分界在数据坐标 k 处；线宽跟着格子边长走
    if boundaries:
        linewidth = max(1.0, cell_size * 72 * 0.15)
        for pos in boundaries:
            ax.axhline(pos, color="white", linewidth=linewidth)
            ax.axvline(pos, color="white", linewidth=linewidth)

    save_figure(fig, out_path, dpi)
    return similarity


def label_size(cell_size: float) -> float:
    """字号跟着格子边长走，避免标签互相压住。"""
    return max(3.5, min(9.0, cell_size * 72 * 0.8))


def report(similarity: np.ndarray, sentences: list[str], top_k: int = 5) -> None:
    """打印相似度统计与最相似的若干句对。"""
    n = len(sentences)
    if n < 2:
        return
    mask = ~np.eye(n, dtype=bool)
    off = similarity[mask]
    print(
        f"相似度（非对角）: min {off.min():.4f} | mean {off.mean():.4f} | max {off.max():.4f}"
    )

    upper = np.triu_indices(n, k=1)
    ranked = np.argsort(similarity[upper])[::-1][:top_k]
    print(f"最相似的 {min(top_k, len(ranked))} 对：")
    for rank, order in enumerate(ranked, 1):
        i, j = int(upper[0][order]), int(upper[1][order])
        print(f"  {rank}. {similarity[i, j]:.4f}  {sentences[i]}  ||  {sentences[j]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="画 LaBSE 句子表征热力图与 text-text 余弦相似度热力图（格子为正方形，分两张 PNG）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subject", help="被试编号，如 sub-01（与 --task 一起选定 CSV）")
    parser.add_argument("--task", choices=("reading", "imagine"), help="任务名，决定用哪张 CSV")
    parser.add_argument("--csv", type=Path, help="直接指定 CSV，给出后忽略 --subject / --task")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="句表 CSV 所在目录")
    parser.add_argument(
        "--types",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="只画这些 type_number 的句子（空格分隔，如 --types 13 19 26）；同类会按参数顺序合并成块",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="图片输出目录")
    parser.add_argument(
        "--embedding-cell-size",
        type=float,
        default=0.05,
        help="embedding 图每个格子的边长（英寸）；768 列，0.05 对应约 38 英寸宽的画布",
    )
    parser.add_argument(
        "--similarity-cell-size",
        type=float,
        default=None,
        help="相似度图每个格子的边长（英寸）；默认 10/N，使整图约 10 英寸见方",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="只画 embedding 图，跳过 N x N 相似度图（句子多时用它避免内存不足）",
    )
    parser.add_argument(
        "--model-name", default="sentence-transformers/LaBSE", help="SentenceTransformer 模型名或本地路径"
    )
    parser.add_argument("--dpi", type=int, default=300, help="保存图片的 dpi")
    parser.add_argument("--top-k", type=int, default=5, help="终端里打印多少对最相似句子")
    args = parser.parse_args(argv)

    if args.types is not None and any(type_id < 0 for type_id in args.types):
        parser.error("--types 里的类别编号必须是非负整数")
    if args.embedding_cell_size <= 0:
        parser.error("--embedding-cell-size 必须为正数")
    if args.similarity_cell_size is not None and args.similarity_cell_size <= 0:
        parser.error("--similarity-cell-size 必须为正数")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_chinese_font()

    csv_path = resolve_csv(args)
    sentences, type_ids, type_names = load_sentences(csv_path, args.types)
    types_desc = "全部类别" if args.types is None else "/".join(map(str, args.types))
    print(f"[Data] {csv_path} -> {len(sentences)} 句（types={types_desc}）")
    report_types(type_ids, type_names)

    encoder = LaBSETextEncoder(model_name=args.model_name)
    embeddings = encoder(sentences).cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"[Model] {args.model_name} embedding={embeddings.shape} "
        f"L2 范数 min/max = {norms.min():.4f}/{norms.max():.4f}"
    )

    name = csv_path.stem if args.csv is not None else f"{args.subject}_{args.task}"
    emb_path = args.out_dir / f"{name}_text_embedding.png"
    sim_path = args.out_dir / f"{name}_text_similarity.png"
    # 相似度图格子边长没指定时，按「整图约 10 英寸见方」反推
    sim_cell = args.similarity_cell_size
    if sim_cell is None:
        sim_cell = min(0.5, SIMILARITY_TARGET_INCHES / len(sentences))
    # 指定了 --types 时句子已按类别合并成块，在相似度图上标出块间白线
    boundaries = block_boundaries(type_ids) if args.types is not None else None

    plot_embedding(embeddings, sentences, emb_path, args.embedding_cell_size, args.dpi)
    print(f"[Output] {emb_path}")

    if args.skip_similarity:
        print("[Skip] --skip-similarity：不画相似度图")
        return 0

    similarity = plot_similarity(
        embeddings, sentences, sim_path, sim_cell, args.dpi, boundaries
    )
    print(f"[Output] {sim_path}")

    report(similarity, sentences, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
