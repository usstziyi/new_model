"""图 8：run-wise 归一化在三种评估协议下的句类准确率。

数据源：
- 无归一化：``outputs/baseline_lda/main_8_2/summary_all.csv``（cv5，逐折均值）
  与 ``outputs/loro_lodo/main_8_2/summary_all.csv``（loro / lodo，池化准确率）
- 归纳式 / 直推式：``outputs/runwise_{inductive,transductive}/main_8_2/<mode>/summary_all.csv``
  （cv5 取逐折均值，loro / lodo 取池化准确率）

输出：``paper/figures/{colors,gray}/fig8_runwise_overall_170mm.{pdf,tif}``
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
STEM = "fig8_runwise_overall_170mm"

PROTOCOLS = (
    ("cv5", "(a)", "随机 5 折"),
    ("loro", "(b)", "留一 run"),
    ("lodo", "(c)", "留一采集日"),
)
# (取值键, 横轴标签, 是否用斜线填充表示 center-only)
CONDITIONS = (
    ("none", "无\n归一化", False),
    ("inductive_zscore", "归纳\nz-score", False),
    ("inductive_centeronly", "归纳\n中心化", True),
    ("transductive_zscore", "直推\nz-score", False),
    ("transductive_centeronly", "直推\n中心化", True),
)
CHANCE = 0.10


def load_values() -> dict[tuple[str, str], pd.Series]:
    """返回 ``{(protocol, condition): 逐被试准确率 Series}``。"""
    values: dict[tuple[str, str], pd.Series] = {}

    baseline_cv = pd.read_csv(
        OUTPUTS / "baseline_lda/main_8_2/summary_all.csv", encoding="utf-8-sig"
    )
    values[("cv5", "none")] = baseline_cv.set_index("subject")["cv_accuracy_mean"]

    baseline_loo = pd.read_csv(
        OUTPUTS / "loro_lodo/main_8_2/summary_all.csv", encoding="utf-8-sig"
    )
    for protocol in ("loro", "lodo"):
        part = baseline_loo[baseline_loo["mode"] == protocol].set_index("subject")
        values[(protocol, "none")] = part["pooled_accuracy"]

    for condition, folder in (
        ("inductive", "runwise_inductive"),
        ("transductive", "runwise_transductive"),
    ):
        for norm in ("zscore", "centeronly"):
            frame = pd.read_csv(
                OUTPUTS / folder / "main_8_2" / norm / "summary_all.csv",
                encoding="utf-8-sig",
            )
            for protocol in ("cv5", "loro", "lodo"):
                part = frame[frame["protocol"] == protocol].set_index("subject")
                column = "fold_mean_accuracy" if protocol == "cv5" else "accuracy"
                values[(protocol, f"{condition}_{norm}")] = part[column]
    return values


def print_table(values) -> None:
    print("protocol  condition                     mean ± SD      n")
    for protocol, _, _ in PROTOCOLS:
        for condition, label, _ in CONDITIONS:
            series = values[(protocol, condition)]
            print(
                f"{protocol:<9}{condition:<28}{series.mean():.4f} ± "
                f"{series.std(ddof=1):.4f}    {len(series)}"
            )


def draw(mode: str, values) -> plt.Figure:
    palette = style.PALETTE[mode]
    fig, axes = plt.subplots(
        1, 3, figsize=(170 * style.MM, 72 * style.MM), layout="constrained"
    )
    positions = np.arange(len(CONDITIONS))

    for ax, (protocol, marker, label) in zip(axes, PROTOCOLS):
        means = [values[(protocol, cond)].mean() for cond, _, _ in CONDITIONS]
        stds = [values[(protocol, cond)].std(ddof=1) for cond, _, _ in CONDITIONS]

        for index, ((condition, _, hatched), mean, std) in enumerate(
            zip(CONDITIONS, means, stds)
        ):
            key = condition.split("_")[0]
            ax.bar(
                index,
                mean,
                width=0.72,
                yerr=std,
                capsize=1.6,
                error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": palette["text"]},
                **style.bar_style(mode, key, hatched),
            )

        ax.axhline(CHANCE, color=palette["zero"], linestyle=(0, (1, 1.4)), linewidth=0.6)
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_ylim(0, 0.27)
        ax.set_yticks(np.arange(0, 0.26, 0.05))
        ax.set_xticks(positions)
        ax.set_xticklabels([label_ for _, label_, _ in CONDITIONS])
        ax.tick_params(axis="x", pad=1.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, alpha=0.25)
        style.panel_label(ax, marker, suffix=label)

    axes[0].set_ylabel("句类准确率")
    axes[0].text(
        3.55,
        CHANCE + 0.006,
        "chance 10%",
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=palette["zero"],
    )

    # 共享图例放整图底部、居中（规范 §6：小倍数不在每个面板重复图例）
    handles = [
        Patch(**style.bar_style(mode, key, hatched), label=label)
        for key, label, hatched in (
            ("none", "无归一化", False),
            ("inductive", "归纳式", False),
            ("transductive", "直推式", False),
            ("inductive", "斜线：仅减均值", True),
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.05, rect=(0, 0.14, 1, 1))
    return fig


def main() -> None:
    style.apply_style()
    values = load_values()
    print_table(values)
    for mode in ("colors", "gray"):
        fig = draw(mode, values)
        style.save_figure(fig, STEM, mode=mode, preview=True)
        plt.close(fig)
    print(f"[Saved] {style.FIGURE_ROOT}/{{colors,gray}}/{STEM}.pdf/.tif")


if __name__ == "__main__":
    main()
