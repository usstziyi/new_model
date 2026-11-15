"""实验 00：数据审计与缓存构建。

运行::

    uv run python plan_auto/exp00_data_audit/audit.py

产出 ``plan_auto/exp00_data_audit/outputs/``：
- ``audit.csv``        每个被试 × 类别的试次数、run 数、句长均值
- ``n_chars.csv``      每个被试 × 类别 × 句长的计数（检查句长混淆）
- ``bandpower.csv``    类别间频带功率的 eta²（看信号里有没有可分的频谱结构）
- ``summary.md``       人读版本
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402

OUT_DIR = PROJECT_ROOT / "plan_auto/exp00_data_audit/outputs"
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "lowgamma": (30.0, 45.0),
}


def bandpower_features(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """对 (N, C, T) 计算每个频带的平均功率，返回 (N, C, n_bands)。"""
    freqs, psd = welch(eeg, fs=sfreq, nperseg=min(256, eeg.shape[-1]), axis=-1)
    out = np.empty((eeg.shape[0], eeg.shape[1], len(BANDS)), dtype=np.float32)
    for i, (low, high) in enumerate(BANDS.values()):
        mask = (freqs >= low) & (freqs < high)
        out[:, :, i] = np.log10(np.maximum(psd[:, :, mask].mean(axis=-1), 1e-12))
    return out


def eta_squared(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """每个特征维度上，类别标签解释的方差比例。"""
    grand = values.mean(axis=0, keepdims=True)
    ss_total = ((values - grand) ** 2).sum(axis=0)
    ss_between = np.zeros(values.shape[1])
    for label in np.unique(labels):
        part = values[labels == label]
        ss_between += len(part) * (part.mean(axis=0) - grand[0]) ** 2
    return np.divide(ss_between, ss_total, out=np.zeros_like(ss_between), where=ss_total > 0)


def length_only_baseline(labels, n_chars, runs, seed=42) -> float:
    """仅用句长分布做分类的交叉验证准确率（句长捷径基线）。"""
    n_classes = int(labels.max()) + 1
    folds = D.grouped_folds(runs, labels, n_splits=5, seed=seed)
    correct = 0
    total = 0
    for train_idx, test_idx in folds:
        # 每类在训练集里各句长的计数 + 拉普拉斯平滑
        table = np.ones((n_classes, 30))
        for label, length in zip(labels[train_idx], n_chars[train_idx]):
            table[label, min(int(length), 29)] += 1
        log_prior = np.log(table.sum(axis=1)) - np.log(table.sum())
        log_lik = np.log(table / table.sum(axis=1, keepdims=True))
        scores = log_prior[None, :] + log_lik[:, n_chars[test_idx]].T
        pred = scores.argmax(axis=1)
        correct += int((pred == labels[test_idx]).sum())
        total += len(test_idx)
    return correct / total


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, length_rows, bp_rows = [], [], []
    class_names = D.load_class_names()

    for subject in D.SUBJECTS:
        D.build_cache(subject)
        cache = D.load_cache(subject)
        eeg = cache["eeg"]
        labels = cache["labels"]
        runs = cache["runs"]
        n_chars = cache["n_chars"]
        sfreq = cache["meta"]["target_sfreq"]

        for label, class_id in enumerate(D.CLASS_IDS):
            mask = labels == label
            rows.append(
                {
                    "subject": subject,
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, ""),
                    "n": int(mask.sum()),
                    "n_runs": int(np.unique(runs[mask]).size),
                    "n_chars_mean": float(n_chars[mask].mean()),
                    "n_chars_std": float(n_chars[mask].std()),
                }
            )
            for length in range(int(n_chars.min()), int(n_chars.max()) + 1):
                length_rows.append(
                    {
                        "subject": subject,
                        "class_id": class_id,
                        "n_chars": length,
                        "count": int(((labels == label) & (n_chars == length)).sum()),
                    }
                )

        # 每个类别抽 40 条算频带功率
        rng = np.random.default_rng(42)
        sample_idx = np.concatenate(
            [rng.choice(np.flatnonzero(labels == c), size=min(40, int((labels == c).sum())), replace=False)
             for c in range(len(D.CLASS_IDS))]
        )
        sample = np.asarray(eeg[sample_idx], dtype=np.float32)
        feats = bandpower_features(sample, sfreq)  # (N, C, bands)
        sample_labels = labels[sample_idx]
        for band_i, band in enumerate(BANDS):
            eta = eta_squared(feats[:, :, band_i], sample_labels)
            top = np.argsort(eta)[-5:][::-1]
            bp_rows.append(
                {
                    "subject": subject,
                    "band": band,
                    "eta2_max": float(eta.max()),
                    "eta2_mean": float(eta.mean()),
                    "top_channels": ",".join(map(str, top)),
                }
            )

        baseline = length_only_baseline(labels, n_chars, runs)
        matched = D.length_matched_indices(labels, n_chars)
        matched_counts = np.bincount(labels[matched], minlength=len(D.CLASS_IDS))
        print(
            f"[Audit] {subject}: n={len(labels)} 句长配平子集 n={len(matched)} "
            f"(每类 {matched_counts.min()}-{matched_counts.max()}) "
            f"句长捷径基线 acc={baseline:.3f}",
            flush=True,
        )
        # 把每个被试的汇总写到单独 json
        (OUT_DIR / f"{subject}_summary.json").write_text(
            json.dumps(
                {
                    "subject": subject,
                    "n_trials": int(len(labels)),
                    "n_runs": int(np.unique(runs).size),
                    "n_days": int(np.unique(cache["days"]).size),
                    "length_only_baseline_acc": baseline,
                    "length_matched_n": int(len(matched)),
                    "length_matched_per_class": matched_counts.tolist(),
                    "duration_s": float(eeg.shape[-1] / sfreq),
                    "n_channels": int(eeg.shape[1]),
                    "sfreq": sfreq,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    pd.DataFrame(rows).to_csv(OUT_DIR / "audit.csv", index=False)
    pd.DataFrame(length_rows).to_csv(OUT_DIR / "n_chars.csv", index=False)
    pd.DataFrame(bp_rows).to_csv(OUT_DIR / "bandpower.csv", index=False)

    audit = pd.DataFrame(rows)
    bp = pd.DataFrame(bp_rows)
    lines = ["# 数据审计", ""]
    for subject in D.SUBJECTS:
        sub = audit[audit.subject == subject]
        summary = json.loads((OUT_DIR / f"{subject}_summary.json").read_text(encoding="utf-8"))
        lines += [
            f"## {subject}",
            "",
            f"- 试次 {summary['n_trials']}，run {summary['n_runs']}，day {summary['n_days']}，"
            f"时长 {summary['duration_s']:.2f}s @ {summary['sfreq']} Hz，通道 {summary['n_channels']}",
            f"- 句长捷径基线（仅用 n_chars，按 run 分组 CV）：{summary['length_only_baseline_acc']:.3f}",
            f"- 句长配平子集：{summary['length_matched_n']} 条 "
            f"（每类 {min(summary['length_matched_per_class'])}-{max(summary['length_matched_per_class'])}）",
        ]
        best = bp[bp.subject == subject].sort_values("eta2_max", ascending=False).head(3)
        lines.append("- 频带可分性（eta² 最大）：" + "；".join(
            f"{r.band}={r.eta2_max:.3f} (ch {r.top_channels})" for r in best.itertuples()
        ))
        lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
