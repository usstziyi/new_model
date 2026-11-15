"""生成报告用图（英文坐标轴标签，避免中文字体缺失）。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FIG = PROJECT_ROOT / "plan_auto/figures"
FIG.mkdir(parents=True, exist_ok=True)


def bar_with_ci(csv_path: Path, out_path: Path, title: str, top_n: int = 12) -> None:
    table = pd.read_csv(csv_path).sort_values("matched_accuracy", ascending=False).head(top_n)
    table = table.iloc[::-1]
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(table) + 1.6))
    err = np.vstack(
        [table.matched_accuracy - table.matched_ci_low, table.matched_ci_high - table.matched_accuracy]
    )
    ax.barh(y, table.matched_accuracy, xerr=err, color="#4C78A8", alpha=0.85, capsize=3)
    ax.axvline(0.10, color="#E45756", linestyle="--", linewidth=1, label="chance (10%)")
    ax.set_yticks(y, table.config.tolist(), fontsize=9)
    ax.set_xlabel("Length-matched accuracy (95% CI, run-level bootstrap)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.08, max(0.16, float(table.matched_ci_high.max()) + 0.01))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def per_subject_plot() -> None:
    imagine = pd.read_csv(PROJECT_ROOT / "plan_auto/final_model/per_subject_imagine.csv")
    reading = pd.read_csv(PROJECT_ROOT / "plan_auto/exp07_reading/outputs/all_config_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (task, table, chance) in zip(
        axes,
        [
            ("Imagine (naive ensemble)", imagine[imagine.candidate == "naive_ens"], 0.10),
            (
                "Reading (band ensemble)",
                reading[reading.config == "band"].rename(columns={"matched_accuracy": "matched_accuracy"}),
                0.10,
            ),
        ],
    ):
        subjects = sorted(table.subject.unique())
        values = [float(table[table.subject == s].matched_accuracy.iloc[0]) for s in subjects]
        ax.bar(subjects, values, color="#72B7B2")
        ax.axhline(chance, color="#E45756", linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 0.20)
        ax.set_title(task, fontsize=11)
        ax.set_ylabel("Length-matched accuracy")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "per_subject.png", dpi=180)
    plt.close(fig)


def confusion_plot() -> None:
    data = np.load(PROJECT_ROOT / "plan_auto/final_model/final_oof.npz")
    y_true, prob, mask = data["y_true"], data["prob"], data["matched_mask"]
    pred = prob.argmax(1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true[mask], pred[mask]):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Final ensemble (length-matched, pooled 5 subjects)")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_final.png", dpi=180)
    plt.close(fig)


def cross_task_plot() -> None:
    table = pd.read_csv(PROJECT_ROOT / "plan_auto/exp08_cross_task/outputs/results.csv")
    order = ["read->imagine_paired", "read->imagine_runsplit", "imagine_cv", "read_cv",
             "imagine->read_runsplit", "imagine->read_paired"]
    order = [o for o in order if o in set(table.setting)]
    means = table.groupby("setting").accuracy.mean().reindex(order)
    stds = table.groupby("setting").accuracy.std().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(range(len(order)), means, yerr=stds, capsize=3, color="#B279A2")
    ax.axhline(0.102, color="#E45756", linestyle="--", linewidth=1, label="chance (10.2%)")
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (mean ± sd over subjects)")
    ax.set_title("Reading ↔ imagining transfer: paired vs. disjoint-run evaluation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cross_task.png", dpi=180)
    plt.close(fig)


def confound_plot() -> None:
    table = pd.read_csv(PROJECT_ROOT / "plan_auto/exp09_confound/outputs/all_results.csv")
    order = ["position_only", "raw|band_ens", "posdetrend|band_ens"]
    labels = ["position only\n(no EEG)", "EEG bandpower\n(raw)", "EEG bandpower\n(position-detrended)"]
    means = [table[table.setting == s].run_balanced_recall.mean() for s in order]
    stds = [table[table.setting == s].run_balanced_recall.std() for s in order]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(range(3), means, yerr=stds, capsize=4,
                  color=["#E45756", "#4C78A8", "#72B7B2"])
    ax.axhline(0.10, color="black", linestyle="--", linewidth=1, label="chance (10%)")
    for rect, value in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.005, f"{value:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_ylabel("Run-balanced recall (mean ± sd over subjects)")
    ax.set_title("Class information is explained by within-run trial position")
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "confound.png", dpi=180)
    plt.close(fig)


def main() -> None:
    bar_with_ci(
        PROJECT_ROOT / "plan_auto/exp06_sweep/outputs/pooled_config_results.csv",
        FIG / "imagine_configs.png",
        "Imagined speech: feature/model sweep (5 subjects, 10 categories)",
    )
    bar_with_ci(
        PROJECT_ROOT / "plan_auto/exp07_reading/outputs/pooled_config_results.csv",
        FIG / "reading_configs.png",
        "Reading: feature/model sweep (5 subjects, 10 categories)",
    )
    per_subject_plot()
    confusion_plot()
    cross_task_plot()
    confound_plot()
    print(f"[Figures] 写入 {FIG}")


if __name__ == "__main__":
    main()
