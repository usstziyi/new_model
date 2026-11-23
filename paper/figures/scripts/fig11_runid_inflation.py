"""图 11：随机划分虚高与 run 指纹可解码性的关系。

数据源：``paper/tables/table_deepnet_subject.csv``。

(a) 每个模型的 ``cv5 − LORO`` 虚高（柱：5 被试均值，误差棒：被试间标准差，
    空心圆：单个被试）。线性基线的虚高约为深度模型的两倍以上。
(b) 横轴为 run 身份可解码性（随机 5 折交叉验证准确率，%），纵轴为 cv5 超出
    chance 的优势（pp）；每个点是一个「被试 × 模型」，空心为 LORO 超出 chance
    的优势。拟合线只针对实心点。若点落在虚线上，说明 cv5 的优势全部可由 run
    指纹解释。

输出：``paper/figures/{colors,gray}/fig11_runid_inflation_170mm.{pdf,tif}``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

TABLE = style.PROJECT_ROOT / "paper" / "tables" / "table_deepnet_subject.csv"
STEM = "fig11_runid_inflation_170mm"
CHANCE = 0.10

MODELS = (
    ("linear", "linear", "线性 LDA"),
    ("eegnet", "eegnet", "EEGNet"),
    ("freqcnn", "freqcnn", "频域 CNN"),
)
MARKERS = {"linear": "o", "eegnet": "s", "freqcnn": "^"}


def load() -> pd.DataFrame:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE} 不存在，先运行 outputs/collect_deepnet_summary.py")
    return pd.read_csv(TABLE, encoding="utf-8-sig")


def print_stats(frame: pd.DataFrame) -> None:
    print("model      inflation_loro(pp)   runid(%)   cv5-chance(pp)   loro-chance(pp)")
    for model, _, label in MODELS:
        part = frame[frame["model"] == model]
        print(
            f"{label:<10}{part['inflation_loro'].mean()*100:8.2f} ± "
            f"{part['inflation_loro'].std(ddof=1)*100:4.2f}"
            f"{part['runid'].mean()*100:11.2f}"
            f"{(part['cv5'].mean()-CHANCE)*100:14.2f}"
            f"{(part['loro'].mean()-CHANCE)*100:16.2f}"
        )
    r, p = pearsonr(frame["runid"], frame["cv5"] - CHANCE)
    print(f"\nrunid vs cv5 超 chance: r = {r:+.3f}, p = {p:.4g}, n = {len(frame)}")
    r2, p2 = pearsonr(frame["runid"], frame["inflation_loro"])
    print(f"runid vs cv5-LORO 虚高 : r = {r2:+.3f}, p = {p2:.4g}")


def draw(mode: str, frame: pd.DataFrame) -> plt.Figure:
    palette = style.PALETTE[mode]
    fig, axes = plt.subplots(
        1, 2, figsize=(170 * style.MM, 72 * style.MM), layout="constrained"
    )
    ax_bar, ax_scatter = axes
    rng = np.random.default_rng(1)

    # ---------- (a) 虚高 ----------
    positions = np.arange(len(MODELS))
    for index, (model, key, _) in enumerate(MODELS):
        values = frame.loc[frame["model"] == model, "inflation_loro"].to_numpy()
        ax_bar.bar(
            index,
            values.mean(),
            width=0.62,
            yerr=values.std(ddof=1),
            capsize=1.8,
            error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": palette["text"]},
            zorder=2,
            **style.bar_style(mode, key),
        )
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax_bar.scatter(
            index + jitter,
            values,
            s=7,
            facecolors="white",
            edgecolors=palette["text"],
            linewidths=0.45,
            zorder=3,
        )
    ax_bar.axhline(0, color=palette["zero"], linestyle="--", linewidth=0.6)
    ax_bar.set_xlim(-0.62, len(MODELS) - 0.38)
    ax_bar.set_ylim(0, 0.135)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([item[2] for item in MODELS])
    ax_bar.tick_params(axis="x", pad=1.5)
    ax_bar.set_ylabel("虚高（cv5 − LORO，pp）")
    ax_bar.set_axisbelow(True)
    ax_bar.grid(axis="y", linewidth=0.35, alpha=0.25)
    style.panel_label(ax_bar, "(a)")

    # ---------- (b) run 指纹 vs 虚高 ----------
    for model, _, label in MODELS:
        part = frame[frame["model"] == model]
        x = part["runid"].to_numpy() * 100
        y = (part["cv5"].to_numpy() - CHANCE) * 100
        y_loro = (part["loro"].to_numpy() - CHANCE) * 100
        face, edge = palette[model]
        ax_scatter.scatter(
            x, y, s=16, marker=MARKERS[model], facecolors=face, edgecolors=edge,
            linewidths=0.5, zorder=4, label=label,
        )
        ax_scatter.scatter(
            x, y_loro, s=16, marker=MARKERS[model], facecolors="white",
            edgecolors=edge, linewidths=0.5, zorder=3,
        )

    x_all = frame["runid"].to_numpy() * 100
    y_all = (frame["cv5"].to_numpy() - CHANCE) * 100
    slope, intercept = np.polyfit(x_all, y_all, 1)
    x_line = np.linspace(x_all.min() - 1, x_all.max() + 1, 50)
    ax_scatter.plot(x_line, slope * x_line + intercept,
                    color=palette["fit"], linewidth=0.8, zorder=2)
    r, _ = pearsonr(x_all, y_all)
    ax_scatter.axhline(0, color=palette["zero"], linestyle="--", linewidth=0.6)
    ax_scatter.set_xlim(x_all.min() - 3, x_all.max() + 3)
    ax_scatter.set_ylim(-2.5, 14.5)
    ax_scatter.set_xlabel("run 身份可解码性（%）")
    ax_scatter.set_ylabel("超出 chance 的优势（pp）")
    ax_scatter.set_axisbelow(True)
    ax_scatter.grid(axis="y", linewidth=0.35, alpha=0.25)
    style.stats_box(ax_scatter, f"r = {r:.2f}", mode=mode)
    ax_scatter.text(
        0.03,
        0.03,
        "实心：cv5\n空心：LORO",
        transform=ax_scatter.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=palette["text"],
    )
    style.panel_label(ax_scatter, "(b)")

    handles = [
        Patch(**style.bar_style(mode, key), label=label)
        for _, key, label in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=2.2,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.05, rect=(0, 0.13, 1, 1))
    return fig


def main() -> None:
    style.apply_style()
    frame = load()
    print_stats(frame)
    for mode in ("colors", "gray"):
        fig = draw(mode, frame)
        style.save_figure(fig, STEM, mode=mode, preview=True)
        plt.close(fig)
    print(f"[Saved] {style.FIGURE_ROOT}/{{colors,gray}}/{STEM}.pdf/.tif")


if __name__ == "__main__":
    main()
