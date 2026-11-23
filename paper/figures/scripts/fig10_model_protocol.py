"""图 10：线性基线与两个深度基线在三种评估协议下的句类准确率。

数据源：

- ``paper/tables/table_deepnet_subject.csv``（由 ``outputs/collect_deepnet_summary.py`` 生成）

三个面板依次为同 run 随机 5 折、留一 run（LORO）、留一采集日（LODO）。柱高为
5 名被试的均值，误差棒为被试间标准差，空心圆为单个被试。三档使用相同的 10 类
句类、相同的 5 折划分（深度模型）或相同的随机种子（线性基线）。

输出：``paper/figures/{colors,gray}/fig10_model_protocol_170mm.{pdf,tif}``
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

TABLE = style.PROJECT_ROOT / "paper" / "tables" / "table_deepnet_subject.csv"
STEM = "fig10_model_protocol_170mm"
CHANCE = 0.10

MODELS = (
    ("linear", "linear", "线性 LDA", "线性\nLDA"),
    ("eegnet", "eegnet", "EEGNet", "EEGNet"),
    ("freqcnn", "freqcnn", "频域 CNN", "频域\nCNN"),
)
PROTOCOLS = (
    ("cv5", "(a)", "随机 5 折"),
    ("loro", "(b)", "留一 run"),
    ("lodo", "(c)", "留一采集日"),
)


def load() -> pd.DataFrame:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE} 不存在，先运行 outputs/collect_deepnet_summary.py")
    return pd.read_csv(TABLE, encoding="utf-8-sig")


def print_table(frame: pd.DataFrame) -> None:
    for protocol, _, label in PROTOCOLS:
        print(f"\n[{protocol}] {label}")
        for model, _, model_label, _ in MODELS:
            part = frame.loc[frame["model"] == model, protocol]
            print(
                f"  {model_label:<8} {part.mean():.4f} ± {part.std(ddof=1):.4f}  "
                f"逐被试 {np.round(part.values * 100, 2).tolist()}"
            )


def draw(mode: str, frame: pd.DataFrame) -> plt.Figure:
    palette = style.PALETTE[mode]
    fig, axes = plt.subplots(
        1, 3, figsize=(170 * style.MM, 72 * style.MM), layout="constrained"
    )
    positions = np.arange(len(MODELS))
    rng = np.random.default_rng(0)

    for ax, (protocol, marker, label) in zip(axes, PROTOCOLS):
        for index, (model, key, _, _) in enumerate(MODELS):
            values = frame.loc[frame["model"] == model, protocol].to_numpy()
            mean = values.mean()
            std = values.std(ddof=1)
            ax.bar(
                index,
                mean,
                width=0.62,
                yerr=std,
                capsize=1.8,
                error_kw={
                    "elinewidth": 0.6,
                    "capthick": 0.6,
                    "ecolor": palette["text"],
                },
                zorder=2,
                **style.bar_style(mode, key),
            )
            jitter = rng.uniform(-0.13, 0.13, size=len(values))
            ax.scatter(
                index + jitter,
                values,
                s=7,
                facecolors="white",
                edgecolors=palette["text"],
                linewidths=0.45,
                zorder=3,
            )

        ax.axhline(CHANCE, color=palette["zero"], linestyle=(0, (1, 1.4)), linewidth=0.6)
        ax.set_xlim(-0.62, len(MODELS) - 0.38)
        ax.set_ylim(0, 0.27)
        ax.set_yticks(np.arange(0, 0.26, 0.05))
        ax.set_xticks(positions)
        ax.set_xticklabels([item[3] for item in MODELS])
        ax.tick_params(axis="x", pad=1.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linewidth=0.35, alpha=0.25)
        style.panel_label(ax, marker, suffix=label)

    axes[0].set_ylabel("句类准确率")

    handles = [
        Patch(**style.bar_style(mode, key), label=model_label)
        for _, key, model_label, _ in MODELS
    ]
    handles.append(
        plt.Line2D(
            [], [], marker="o", linestyle="none", markersize=3,
            markerfacecolor="white", markeredgecolor=palette["text"],
            markeredgewidth=0.45, label="单个被试",
        )
    )
    handles.append(
        plt.Line2D([], [], color=palette["zero"], linestyle=(0, (1, 1.4)),
                   linewidth=0.6, label="机会水平 10%")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.05, rect=(0, 0.13, 1, 1))
    return fig


def main() -> None:
    style.apply_style()
    frame = load()
    print_table(frame)
    for mode in ("colors", "gray"):
        fig = draw(mode, frame)
        style.save_figure(fig, STEM, mode=mode, preview=True)
        plt.close(fig)
    print(f"[Saved] {style.FIGURE_ROOT}/{{colors,gray}}/{STEM}.pdf/.tif")


if __name__ == "__main__":
    main()
