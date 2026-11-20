"""图 9：run-wise 归一化的逐 run 损失与无归一化准确率的关系。

数据源：``outputs/runwise_paired/paired_zscore_loro.csv``（5 被试共 224 个 run）。

(a) 每个 run 在无归一化下的准确率（横轴）与归一化后的变化（纵轴）的散点，
    配线性拟合；差值 = 归一化后 − 无归一化，负值表示变差。
(b) 按无归一化准确率五分位分组的平均差值，归纳式与直推式并列。

输出：``paper/figures/{colors,gray}/fig9_runwise_paired_170mm.{pdf,tif}``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

OUTPUTS = style.PROJECT_ROOT / "outputs"
STEM = "fig9_runwise_paired_170mm"
CSV = OUTPUTS / "runwise_paired/paired_zscore_loro.csv"

PAIRS = (
    ("delta_inductive_minus_baseline", "inductive", "归纳式"),
    ("delta_transductive_minus_baseline", "transductive", "直推式"),
)


def load() -> pd.DataFrame:
    frame = pd.read_csv(CSV, encoding="utf-8-sig")
    if frame.empty:
        raise SystemExit(f"{CSV} 为空，先跑 outputs/collect_runwise_paired.py")
    return frame


def add_quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["quintile"] = pd.qcut(frame["baseline_accuracy"], 5, labels=False)
    return frame


def print_table(frame: pd.DataFrame) -> None:
    print(f"n_runs = {len(frame)}")
    for column, _, label in PAIRS:
        r = np.corrcoef(frame["baseline_accuracy"], frame[column])[0, 1]
        print(f"{label:<6} mean_delta={frame[column].mean():+.4f}  pearson_r={r:+.3f}")
    print("\nquintile  n   baseline_mean   inductive    transductive")
    for quintile, part in frame.groupby("quintile"):
        print(
            f"Q{int(quintile) + 1:<8}{len(part):<4}{part['baseline_accuracy'].mean():.4f}"
            f"{part[PAIRS[0][0]].mean():>16.4f}{part[PAIRS[1][0]].mean():>14.4f}"
        )


def draw(mode: str, frame: pd.DataFrame) -> plt.Figure:
    palette = style.PALETTE[mode]
    fig, axes = plt.subplots(
        1, 2, figsize=(170 * style.MM, 72 * style.MM), layout="constrained"
    )
    ax_scatter, ax_bars = axes

    # ---------- (a) 散点 + 线性拟合 ----------
    x = frame["baseline_accuracy"].to_numpy()
    y = frame[PAIRS[0][0]].to_numpy()
    ax_scatter.axhline(0, color=palette["zero"], linestyle="--", linewidth=0.6)
    ax_scatter.axvline(0.10, color=palette["zero"], linestyle=(0, (1, 1.4)), linewidth=0.6)
    ax_scatter.scatter(
        x,
        y,
        s=9,
        facecolors="none",
        edgecolors=palette["marker"],
        linewidths=0.4,
        alpha=0.75,
        zorder=3,
    )
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 50)
    ax_scatter.plot(
        x_line, slope * x_line + intercept, color=palette["fit"], linewidth=0.8, zorder=4
    )

    r_inductive = np.corrcoef(x, y)[0, 1]
    r_transductive = np.corrcoef(x, frame[PAIRS[1][0]].to_numpy())[0, 1]
    style.stats_box(
        ax_scatter,
        f"r = {r_inductive:.2f}（归纳）\nr = {r_transductive:.2f}（直推）",
        mode=mode,
    )
    ax_scatter.set_xlim(0, 0.45)
    y_pad = 0.03
    ax_scatter.set_ylim(y.min() - y_pad, y.max() + y_pad)
    ax_scatter.set_xlabel("无归一化逐 run 准确率")
    ax_scatter.set_ylabel("Δ 准确率（归一化后 − 无归一化）")
    ax_scatter.set_axisbelow(True)
    ax_scatter.grid(axis="y", linewidth=0.35, alpha=0.25)
    style.panel_label(ax_scatter, "(a)")

    # ---------- (b) 五分位平均差 ----------
    quintiles = sorted(frame["quintile"].dropna().unique())
    positions = np.arange(len(quintiles))
    width = 0.36
    means_by_series: list[list[float]] = []
    for offset, (column, key, _) in zip((-width / 2, width / 2), PAIRS):
        means = [
            frame.loc[frame["quintile"] == quintile, column].mean()
            for quintile in quintiles
        ]
        means_by_series.append(means)
        ax_bars.bar(
            positions + offset,
            means,
            width=width * 0.92,
            **style.bar_style(mode, key),
        )

    ax_bars.axhline(0, color=palette["zero"], linestyle="--", linewidth=0.6)
    ax_bars.set_xticks(positions)
    ax_bars.set_xticklabels([f"Q{int(q) + 1}" for q in quintiles])
    ax_bars.set_xlim(-0.6, len(quintiles) - 0.4)
    flat = np.asarray(means_by_series)
    ax_bars.set_ylim(flat.min() - 0.025, flat.max() + 0.045)
    ax_bars.set_xlabel("无归一化准确率五分位（低 → 高）")
    ax_bars.set_ylabel("Δ 准确率")
    ax_bars.set_axisbelow(True)
    ax_bars.grid(axis="y", linewidth=0.35, alpha=0.25)
    style.panel_label(ax_bars, "(b)")

    handles = [
        Patch(**style.bar_style(mode, key), label=label) for _, key, label in PAIRS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=2.0,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.05, rect=(0, 0.12, 1, 1))
    return fig


def main() -> None:
    style.apply_style()
    frame = add_quintiles(load())
    print_table(frame)
    for mode in ("colors", "gray"):
        fig = draw(mode, frame)
        style.save_figure(fig, STEM, mode=mode, preview=True)
        plt.close(fig)
    print(f"[Saved] {style.FIGURE_ROOT}/{{colors,gray}}/{STEM}.pdf/.tif")


if __name__ == "__main__":
    main()
