"""图 12：逐 run 比较深度基线与线性基线（LORO 协议）。

数据源：

- 线性基线：``outputs/loro_lodo/main_8_2/sub-*/paired_loro_*.csv``（``loo_accuracy``）
- EEGNet：``outputs/deepnet/eegnet/main_eegnet/sub-*/folds_loro_*.csv``
- 频域 CNN：``outputs/deepnet/freqcnn/main_freqcnn/sub-*/folds_loro_*.csv``

每个点是一个留出的 run（5 名被试共 224 个 run）。横轴为线性基线的留一准确率，
纵轴为深度模型的留一准确率，虚线为 ``y = x``。点越贴近虚线，说明两种表示在
严格协议下越接近；面板内标注 Pearson r、平均差与变好/变差 run 数。

输出：``paper/figures/{colors,gray}/fig12_deep_vs_linear_paired_170mm.{pdf,tif}``
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

OUTPUTS = style.PROJECT_ROOT / "outputs"
STEM = "fig12_deep_vs_linear_paired_170mm"
CHANCE = 0.10

PANELS = (
    ("(a)", "eegnet", "EEGNet"),
    ("(b)", "freqcnn", "频域 CNN"),
)


def load_linear() -> pd.DataFrame:
    parts = []
    for path in sorted(glob.glob(str(OUTPUTS / "loro_lodo/main_8_2/sub-*/paired_loro_*.csv"))):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        parts.append(frame[["subject", "group", "loo_accuracy"]].rename(columns={"group": "split"}))
    if not parts:
        raise SystemExit("缺少线性基线的 paired_loro_*.csv")
    return pd.concat(parts, ignore_index=True)


def load_deep(experiment: str, run: str) -> pd.DataFrame:
    parts = []
    for path in sorted(glob.glob(str(OUTPUTS / "deepnet" / experiment / run / "sub-*/folds_loro_*.csv"))):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        parts.append(frame[["subject", "split", "accuracy"]])
    if not parts:
        raise SystemExit(f"缺少 {experiment} 的 folds_loro_*.csv")
    return pd.concat(parts, ignore_index=True)


def print_stats(data: dict[str, pd.DataFrame]) -> None:
    for _, key, label in PANELS:
        frame = data[key]
        diff = (frame["deep"] - frame["linear"]) * 100
        r, p = pearsonr(frame["linear"], frame["deep"])
        print(
            f"{label:<8} n={len(frame)}  r={r:+.3f} (p={p:.2g})  "
            f"mean Δ={diff.mean():+.2f}pp  Wilcoxon p={wilcoxon(frame['deep'], frame['linear']).pvalue:.3f}  "
            f"变好/变差={int((diff > 0).sum())}/{int((diff < 0).sum())}"
        )


def draw(mode: str, data: dict[str, pd.DataFrame]) -> plt.Figure:
    palette = style.PALETTE[mode]
    fig, axes = plt.subplots(
        1, 2, figsize=(170 * style.MM, 72 * style.MM), layout="constrained"
    )

    for ax, (marker, key, label) in zip(axes, PANELS):
        frame = data[key]
        face, edge = palette[key]
        ax.plot([0, 0.34], [0, 0.34], color=palette["zero"],
                linestyle="--", linewidth=0.6, zorder=1)
        ax.axvline(CHANCE, color=palette["zero"], linestyle=(0, (1, 1.4)),
                   linewidth=0.5, zorder=1)
        ax.axhline(CHANCE, color=palette["zero"], linestyle=(0, (1, 1.4)),
                   linewidth=0.5, zorder=1)
        ax.scatter(
            frame["linear"], frame["deep"], s=7, facecolors=face, edgecolors=edge,
            linewidths=0.4, alpha=0.8, zorder=3,
        )

        diff = (frame["deep"] - frame["linear"]) * 100
        r = np.corrcoef(frame["linear"], frame["deep"])[0, 1]
        mean_lin = frame["linear"].mean() * 100
        mean_deep = frame["deep"].mean() * 100
        style.stats_box(
            ax,
            f"r = {r:.2f}\n均值 {mean_lin:.1f}% → {mean_deep:.1f}%\n"
            f"Δ 均值 {diff.mean():+.2f}pp",
            mode=mode,
        )
        ax.set_xlim(0, 0.34)
        ax.set_ylim(0, 0.34)
        ax.set_xticks(np.arange(0, 0.31, 0.1))
        ax.set_yticks(np.arange(0, 0.31, 0.1))
        ax.set_xlabel("线性 LDA 留一准确率")
        ax.set_ylabel(f"{label} 留一准确率")
        ax.set_aspect("equal")
        ax.set_axisbelow(True)
        ax.grid(linewidth=0.3, alpha=0.2)
        style.panel_label(ax, marker, suffix=label)

    handles = [
        plt.Line2D([], [], color=palette["zero"], linestyle="--", linewidth=0.6, label="y = x"),
        plt.Line2D([], [], color=palette["zero"], linestyle=(0, (1, 1.4)),
                   linewidth=0.6, label="机会水平 10%"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=2.4,
        handletextpad=0.5,
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.05, rect=(0, 0.11, 1, 1))
    return fig


def main() -> None:
    style.apply_style()
    linear = load_linear()
    data: dict[str, pd.DataFrame] = {}
    for _, key, _ in PANELS:
        experiment = "eegnet" if key == "eegnet" else "freqcnn"
        run = "main_eegnet" if key == "eegnet" else "main_freqcnn"
        deep = load_deep(experiment, run).rename(columns={"accuracy": "deep"})
        merged = linear.merge(deep, on=["subject", "split"], how="inner")
        if merged.empty:
            raise SystemExit(f"{key} 与线性基线没有任何可配对的 run")
        merged = merged.rename(columns={"loo_accuracy": "linear"})
        data[key] = merged
    print_stats(data)
    for mode in ("colors", "gray"):
        fig = draw(mode, data)
        style.save_figure(fig, STEM, mode=mode, preview=True)
        plt.close(fig)
    print(f"[Saved] {style.FIGURE_ROOT}/{{colors,gray}}/{STEM}.pdf/.tif")


if __name__ == "__main__":
    main()
