"""论文配图的统一风格、配色与导出。

对应 ``paper/figures/standards/投稿配图规范.md``：同一份绘图代码分别用
``colors`` 与 ``gray`` 两套调色板渲染，输出 PDF（矢量）+ TIFF（600 dpi、LZW）。

用法::

    from style import MM, PALETTE, apply_style, bar_style, panel_label, save_figure

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(170 * MM, 72 * MM), layout="constrained")
    ...
    save_figure(fig, "fig8_runwise_overall_170mm", mode="colors")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (需在 use("Agg") 之后导入)

MM = 1 / 25.4
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = PROJECT_ROOT / "paper" / "figures"

# 彩色 / 灰度两套编码，键名一致；柱形项为 (填充色, 描边色)
PALETTE = {
    "colors": {
        "none": ("#D9D9D9", "#595959"),
        "inductive": ("#CFDDEC", "#4A6B8A"),
        "transductive": ("#F6D5C0", "#E1812C"),
        # 深度基线对比图：线性 / EEGNet / 频域 CNN
        "linear": ("#D9D9D9", "#595959"),
        "eegnet": ("#CFDDEC", "#4A6B8A"),
        "freqcnn": ("#F6D5C0", "#E1812C"),
        "marker": "#2E6DA4",
        "fit": "#262626",
        "zero": "#7F7F7F",
        "text": "#262626",
    },
    "gray": {
        "none": ("#D9D9D9", "#595959"),
        "inductive": ("#BFBFBF", "#262626"),
        "transductive": ("#737373", "#262626"),
        "linear": ("#E8E8E8", "#595959"),
        "eegnet": ("#BFBFBF", "#262626"),
        "freqcnn": ("#737373", "#262626"),
        "marker": "#262626",
        "fit": "#262626",
        "zero": "#8C8C8C",
        "text": "#262626",
    },
}
CENTER_HATCH = "////"


def apply_style() -> None:
    """套用投稿尺寸下的字号、线宽与字体设置。"""
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],  # 列表才会回退中文
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            # 规范写 6.2，但本环境下含中文的文字低于 6.5 pt 会被静默丢弃
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "lines.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.25,
            "hatch.linewidth": 0.5,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def bar_style(mode: str, key: str, hatched: bool = False) -> dict:
    """返回 ``bar`` 的样式参数；``hatched`` 用于区分"仅中心化"口径。"""
    face, edge = PALETTE[mode][key]
    return {
        "facecolor": face,
        "edgecolor": edge,
        "linewidth": 0.5,
        "hatch": CENTER_HATCH if hatched else None,
    }


def panel_label(
    ax,
    marker: str,
    suffix: str | None = None,
    x: float = 0.0,
    y: float = 1.02,
) -> None:
    """在坐标区左上角外侧写 ``(a)`` 等面板编号，不加图内标题。

    编号本身用 Times New Roman 粗体（宋体没有粗体字重）；``suffix`` 若含中文，
    用默认字体列表渲染，交给 matplotlib 回退到 SimSun。
    """
    ax.text(
        x,
        y,
        marker,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        fontfamily="Times New Roman",
    )
    if suffix:
        ax.text(
            x + 0.115,
            y,
            suffix,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
        )


def stats_box(ax, text: str, mode: str, x: float = 0.97, y: float = 0.96, fontsize: float = 6.5) -> None:
    """面板右上角的统计标注：半透明白底，避免压住数据（规范 §8）。"""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=PALETTE[mode]["text"],
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    )


def save_figure(fig, stem: str, mode: str, preview: bool = False) -> None:
    """把图保存为 ``pdf`` 与 600 dpi ``tif``。

    Args:
        fig: matplotlib figure。
        stem: 不含扩展名的文件名，如 ``fig8_runwise_overall_170mm``。
        mode: ``colors`` 或 ``gray``，决定写到哪个子目录。
        preview: 为 ``True`` 时额外在 ``paper/figures/.preview`` 下写 200 dpi PNG，
            便于快速检查排版（不要交付该文件）。
    """
    out_dir = FIGURE_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        out_dir / f"{stem}.tif",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    if preview:
        preview_dir = FIGURE_ROOT / ".preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            preview_dir / f"{stem}_{mode}.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.02,
        )
