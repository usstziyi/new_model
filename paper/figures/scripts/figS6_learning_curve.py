"""图 S6：EEGNet 在 sub-01 上的学习曲线（训练集 vs 测试集）。

数据来源：``outputs/deepnet/eegnet/curve100_holdout/sub-01/curve_holdout_sub-01.csv``
（8:2 固定划分，100 epoch，每轮在 eval 模式下评估）。

用途：回答"30 epoch 是否训练不足"。曲线显示测试准确率在约 20–25 epoch 后即进入平台，
此后训练准确率继续上升到 86%，但测试不再改善，属于过拟合而非欠拟合。

用法::

    uv run paper/figures/scripts/figS6_learning_curve.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from style import MM, PALETTE, apply_style, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURVE_CSV = (
    PROJECT_ROOT
    / "outputs/deepnet/eegnet/curve100_holdout/sub-01/curve_holdout_sub-01.csv"
)
STEM = "figS6_learning_curve_80mm"
DEFAULT_EPOCHS = 30
# 曲线的线色：彩色版用蓝/橙，灰度版用深/中灰（配合实线/虚线，去色后仍可区分）
LINE_COLORS = {
    "colors": {"train": "#4A6B8A", "test": "#E1812C"},
    "gray": {"train": "#262626", "test": "#8C8C8C"},
}


def draw(mode: str) -> None:
    curve = pd.read_csv(CURVE_CSV)
    apply_style()

    fig, ax = plt.subplots(figsize=(80 * MM, 62 * MM), layout="constrained")

    colors = LINE_COLORS[mode]

    ax.plot(
        curve.epoch,
        curve.train_acc_eval * 100,
        color=colors["train"],
        linewidth=0.9,
        label="训练集",
    )
    ax.plot(
        curve.epoch,
        curve.test_acc_eval * 100,
        color=colors["test"],
        linewidth=0.9,
        linestyle="--",
        label="测试集",
    )

    chance = 10.0
    ax.axhline(chance, color=PALETTE[mode]["zero"], linewidth=0.5, linestyle=":")
    ax.text(
        46,
        chance + 1.4,
        "机会水平 10%",
        fontsize=6.5,
        color=PALETTE[mode]["text"],
        ha="left",
        va="bottom",
    )

    ax.axvline(
        DEFAULT_EPOCHS,
        color=PALETTE[mode]["zero"],
        linewidth=0.5,
        linestyle="-.",
    )
    ax.text(
        DEFAULT_EPOCHS + 1.5,
        30,
        f"默认 {DEFAULT_EPOCHS} epoch",
        fontsize=6.5,
        color=PALETTE[mode]["text"],
        ha="left",
        va="center",
    )

    test_at_default = float(
        curve.loc[curve.epoch == DEFAULT_EPOCHS, "test_acc_eval"].iloc[0]
    ) * 100
    test_final = float(curve.test_acc_eval.iloc[-1]) * 100
    ax.annotate(
        f"{test_final:.1f}%",
        xy=(curve.epoch.iloc[-1], test_final),
        xytext=(-4, -10),
        textcoords="offset points",
        fontsize=6.5,
        color=PALETTE[mode]["text"],
        ha="right",
        va="center",
    )

    ax.set_xlabel("训练轮数（epoch）")
    ax.set_ylabel("准确率（%）")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 95)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y")
    ax.legend(
        loc="center right",
        frameon=False,
        handlelength=1.6,
        labelspacing=0.3,
    )

    print(
        f"[{mode}] epoch {DEFAULT_EPOCHS} test={test_at_default:.2f}% | "
        f"epoch 100 test={test_final:.2f}% | "
        f"epoch 100 train={float(curve.train_acc_eval.iloc[-1])*100:.2f}%"
    )
    save_figure(fig, STEM, mode, preview=True)
    plt.close(fig)


def main() -> None:
    if not CURVE_CSV.exists():
        raise SystemExit(f"缺少学习曲线数据：{CURVE_CSV}")
    for mode in ("colors", "gray"):
        draw(mode)


if __name__ == "__main__":
    main()
