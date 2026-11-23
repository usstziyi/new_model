"""图 S6：EEGNet 学习曲线（5 名被试 + 跨被试汇总）。

数据来源：``outputs/deepnet/eegnet/curve60_holdout/sub-*/curve_holdout_*.csv``
（每被试 8:2 固定划分，60 epoch，每轮在 eval 模式下评估训练集与测试集）。

用途：回答"30 epoch 是否训练不足"。各被试测试准确率在早期即进入平台，
此后训练准确率继续上升而测试不再改善，属于过拟合而非欠拟合。

用法::

    uv run paper/figures/scripts/figS6_learning_curve.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import MM, PALETTE, apply_style, panel_label, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURVE_ROOT = PROJECT_ROOT / "outputs/deepnet/eegnet/curve60_holdout"
STEM = "figS6_learning_curve_170mm"
DEFAULT_EPOCHS = 30
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
# 曲线线色：彩色版用蓝/橙，灰度版用深/中灰（配合实线/虚线，去色后仍可区分）
LINE_COLORS = {
    "colors": {"train": "#4A6B8A", "test": "#E1812C", "band": "#F6D5C0"},
    "gray": {"train": "#262626", "test": "#8C8C8C", "band": "#E8E8E8"},
}
Y_MAX = 95


def load_curves() -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    for subject in SUBJECTS:
        path = CURVE_ROOT / subject / f"curve_holdout_{subject}.csv"
        if not path.exists():
            raise SystemExit(f"缺少学习曲线数据：{path}")
        curves[subject] = pd.read_csv(path).sort_values("epoch")
    return curves


def draw_subject(ax, curve: pd.DataFrame, subject: str, mode: str, marker: str) -> None:
    colors = LINE_COLORS[mode]
    ax.plot(
        curve.epoch,
        curve.train_acc_eval * 100,
        color=colors["train"],
        linewidth=0.8,
    )
    ax.plot(
        curve.epoch,
        curve.test_acc_eval * 100,
        color=colors["test"],
        linewidth=0.8,
        linestyle="--",
    )
    ax.axhline(10, color=PALETTE[mode]["zero"], linewidth=0.45, linestyle=":")
    ax.axvline(
        DEFAULT_EPOCHS, color=PALETTE[mode]["zero"], linewidth=0.45, linestyle="-."
    )
    ax.set_xlim(1, curve.epoch.max())
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks([1, 20, 40, 60])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(True, axis="y")
    panel_label(ax, marker, suffix=subject, x=0.0, y=1.01)


def draw_summary(ax, curves: dict[str, pd.DataFrame], mode: str, marker: str) -> None:
    colors = LINE_COLORS[mode]
    frame = pd.concat(
        [c.assign(subject=s)[["epoch", "test_acc_eval", "train_acc_eval"]] for s, c in curves.items()]
    )
    stats = frame.groupby("epoch")[["test_acc_eval", "train_acc_eval"]].agg(["mean", "std"])
    epochs = stats.index.to_numpy()
    test_mean = stats[("test_acc_eval", "mean")].to_numpy() * 100
    test_sd = stats[("test_acc_eval", "std")].to_numpy() * 100
    train_mean = stats[("train_acc_eval", "mean")].to_numpy() * 100
    train_sd = stats[("train_acc_eval", "std")].to_numpy() * 100

    ax.fill_between(epochs, train_mean - train_sd, train_mean + train_sd,
                    color=colors["band"], linewidth=0, alpha=0.9)
    ax.fill_between(epochs, test_mean - test_sd, test_mean + test_sd,
                    color=colors["band"], linewidth=0, alpha=0.9)
    ax.plot(epochs, train_mean, color=colors["train"], linewidth=1.0, label="训练集")
    ax.plot(epochs, test_mean, color=colors["test"], linewidth=1.0, linestyle="--",
            label="测试集")
    ax.axhline(10, color=PALETTE[mode]["zero"], linewidth=0.45, linestyle=":")
    ax.axvline(DEFAULT_EPOCHS, color=PALETTE[mode]["zero"], linewidth=0.45, linestyle="-.")

    ax.set_xlim(1, epochs.max())
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks([1, 20, 40, 60])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(True, axis="y")
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
        frameon=False,
        handlelength=1.6,
        labelspacing=0.3,
    )
    panel_label(ax, marker, suffix="被试均值", x=0.0, y=1.01)


def draw(mode: str) -> None:
    curves = load_curves()
    apply_style()

    fig, axes = plt.subplots(2, 3, figsize=(170 * MM, 112 * MM), layout="constrained")
    flat = axes.ravel()
    markers = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for ax, subject, marker in zip(flat, SUBJECTS, markers):
        draw_subject(ax, curves[subject], subject, mode, marker)
    draw_summary(flat[5], curves, mode, "(f)")

    for ax in flat[-3:]:
        ax.set_xlabel("训练轮数（epoch）")
    for ax in (flat[0], flat[3]):
        ax.set_ylabel("准确率（%）")

    test30 = np.mean([c.loc[c.epoch == DEFAULT_EPOCHS, "test_acc_eval"].iloc[0] for c in curves.values()]) * 100
    test60 = np.mean([c.test_acc_eval.iloc[-1] for c in curves.values()]) * 100
    print(f"[{mode}] 5 被试 test@{DEFAULT_EPOCHS}={test30:.2f}%  test@60={test60:.2f}%")
    save_figure(fig, STEM, mode, preview=True)
    plt.close(fig)


def main() -> None:
    for mode in ("colors", "gray"):
        draw(mode)


if __name__ == "__main__":
    main()
