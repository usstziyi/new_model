"""论文用图：半合成注入曲线 + 训练级置换零分布。"""

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


def injection_curve() -> None:
    path = PROJECT_ROOT / "plan_auto/exp11_injection/outputs/pooled_curve.csv"
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, variant, title in zip(
        axes,
        ("raw", "posdetrend"),
        ("Raw features", "Position-detrended features"),
    ):
        part = table[table.variant == variant]
        for feature, color, marker in (
            ("bp", "#4C78A8", "o"),
            ("hjorth", "#F58518", "s"),
            ("erp", "#54A24B", "^"),
        ):
            sub = part[part.feature == feature].sort_values("snr")
            ax.plot(sub.snr, sub.accuracy_mean, marker=marker, color=color, label=feature)
            ax.fill_between(sub.snr, sub.accuracy_mean - sub.accuracy_sd,
                            sub.accuracy_mean + sub.accuracy_sd, color=color, alpha=0.15)
        ax.axhline(0.10, color="#E45756", linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("Injected SNR")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.05, 0.75)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Accuracy (mean ± sd over 5 subjects)")
    fig.suptitle("Semi-synthetic injection: pipeline sensitivity vs. effect size", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "injection_curve.png", dpi=180)
    plt.close(fig)


def permutation_null() -> None:
    import json

    root = PROJECT_ROOT / "plan_auto/exp12_permutation/outputs"
    subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (key, title) in zip(
        axes,
        (("accuracy", "Accuracy"), ("run_balanced_recall", "Run-balanced recall")),
    ):
        for variant, color in (("raw", "#4C78A8"), ("posdetrend", "#72B7B2")):
            nulls, observed = [], []
            for subject in subjects:
                data = np.load(root / subject / "permutation.npz")
                summary = json.loads((root / subject / "summary.json").read_text(encoding="utf-8"))
                nulls.append(data[f"{variant}_null_{key}"])
                observed.append(summary[variant]["observed"][key])
            null = np.concatenate(nulls)
            ax.hist(null, bins=30, alpha=0.45, color=color, density=True,
                    label=f"{variant} null (mean={null.mean():.3f})")
            ax.axvline(float(np.mean(observed)), color=color, linestyle="--", linewidth=1.5,
                       label=f"{variant} observed={np.mean(observed):.3f}")
        ax.axvline(0.10, color="#E45756", linestyle=":", linewidth=1, label="chance")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle("Training-level permutation (labels shuffled within run, 300 perms/subject)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "permutation_null.png", dpi=180)
    plt.close(fig)


def main() -> None:
    injection_curve()
    permutation_null()
    print(f"[Figures] 写入 {FIG}")


if __name__ == "__main__":
    main()
