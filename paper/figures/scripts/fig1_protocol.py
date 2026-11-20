"""图 1：三种划分协议下的句类准确率。

数据源：
- ``outputs/baseline_lda/main_8_2/summary_all.csv``（同 run 随机 5 折，逐折均值）
- ``outputs/loro_lodo/main_8_2/summary_all.csv``（LORO / LODO，池化准确率）

本脚本按原图复刻；唯一改动是把 "chance 10%" 标注移到坐标区右侧空白处，
避免被第一根柱子遮挡。

输出：``paper/figures/{colors,gray}/fig1_protocol_170mm.{pdf,tif}``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

OUTPUTS = style.PROJECT_ROOT / "outputs"
STEM = "fig1_protocol_170mm"
CHANCE = 0.10
LABELS = ("同 run 随机 5 折", "留一 run（LORO）", "留一采集日（LODO）")


def load_series() -> list[pd.Series]:
    baseline = pd.read_csv(
        OUTPUTS / "baseline_lda/main_8_2/summary_all.csv", encoding="utf-8-sig"
    ).set_index("subject")["cv_accuracy_mean"]
    loo = pd.read_csv(
        OUTPUTS / "loro_lodo/main_8_2/summary_all.csv", encoding="utf-8-sig"
    )
    loro = loo[loo["mode"] == "loro"].set_index("subject")["pooled_accuracy"]
    lodo = loo[loo["mode"] == "lodo"].set_index("subject")["pooled_accuracy"]
    subjects = sorted(set(baseline.index) & set(loro.index) & set(lodo.index))
    return [series.loc[subjects] for series in (baseline, loro, lodo)]


def draw(mode: str, series: list[pd.Series]) -> plt.Figure:
    palette = style.PALETTE[mode]
    fig, ax = plt.subplots(
        1, 1, figsize=(170 * style.MM, 64 * style.MM), layout="constrained"
    )

    rng = np.random.default_rng(0)
    for index, values in enumerate(series):
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        ax.bar(index, mean, width=0.75, **style.bar_style(mode, "inductive"))
        ax.errorbar(
            index,
            mean,
            yerr=sd,
            color=palette["text"],
            linewidth=0.8,
            capsize=2.5,
            capthick=0.8,
            zorder=4,
        )
        ax.scatter(
            index + rng.normal(0, 0.035, size=len(values)),
            values.to_numpy(),
            s=9,
            facecolors="none",
            edgecolors=palette["marker"],
            linewidths=0.6,
            zorder=5,
        )
        ax.text(
            index,
            mean + sd + 0.008,
            f"{mean * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color=palette["text"],
        )

    ax.axhline(CHANCE, color=palette["zero"], linestyle=(0, (1, 1.4)), linewidth=0.6)
    # 标注放在右侧空白处，避免压住第一根柱子
    ax.text(
        2.96,
        CHANCE + 0.004,
        "chance 10%",
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=palette["zero"],
    )

    ax.set_xlim(-0.6, 3.0)
    ax.set_ylim(0, 0.30)
    ax.set_yticks(np.arange(0, 0.31, 0.05))
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("10 类句类准确率")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.35, alpha=0.25)

    handles = [
        Patch(**style.bar_style(mode, "inductive"), label="被试均值"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=palette["marker"],
            label="单个被试",
        ),
        Line2D([], [], color=palette["text"], linewidth=0.8, label="标准差"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor="#D9D9D9",
        handlelength=1.4,
        borderpad=0.4,
        labelspacing=0.35,
    )
    return fig


def main() -> None:
    style.apply_style()
    series = load_series()
    for values, label in zip(series, LABELS):
        print(f"{label:<18}{values.mean():.4f} ± {values.std(ddof=1):.4f}  n={len(values)}")
    for mode in ("colors", "gray"):
        fig = draw(mode, series)
        style.save_figure(fig, STEM, mode=mode, preview=True)
        plt.close(fig)
    print(f"[Saved] {style.FIGURE_ROOT}/{{colors,gray}}/{STEM}.pdf/.tif")


if __name__ == "__main__":
    main()
