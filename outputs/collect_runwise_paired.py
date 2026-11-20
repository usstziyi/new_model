"""逐 run 配对比较：run-wise 归一化 vs 无归一化。

把三组实验结果按 ``(subject, run/day)`` 对齐，输出每个留出组的准确率变化：

- 无归一化：``outputs/loro_lodo/<run>/<subject>/paired_{loro,lodo}_<subject>.csv``
  里的 ``loo_accuracy``；
- 归纳式：``outputs/runwise_inductive/<run>/<subject>/folds_{loro,lodo}_<subject>.csv``
  里的 ``accuracy``；
- 直推式：``outputs/runwise_transductive/<run>/<subject>/folds_{loro,lodo}_<subject>.csv``
  里的 ``accuracy``。

输出（默认 ``outputs/runwise_paired/``）：

- ``paired_<protocol>.csv``：逐组宽表，含三种口径与两个差值；
- ``paired_<protocol>_summary.csv``：逐被试与跨被试的配对统计——均值差、
  按 run 重采样的 95% CI、Wilcoxon 符号秩 p、变好/持平/变坏的组数。

差值的符号统一为 **归一化后 − 无归一化**：负值表示归一化让该 run 变差。

用法::

    uv run outputs/collect_runwise_paired.py \
        --baseline-run outputs/loro_lodo/main_8_2 \
        --inductive-run outputs/runwise_inductive/main_8_2 \
        --transductive-run outputs/runwise_transductive/main_8_2

    uv run outputs/collect_runwise_paired.py --protocol lodo

不指定 ``*-run`` 时自动选择对应实验目录下最新的、含逐折文件的运行目录；
``_smoke`` / ``_test`` 目录会被跳过，显式传入时不受此限制。

注意：跨被试那一行只有 5 个配对值，Wilcoxon 的最小可能 p 约为 0.0625；
若同时在多个被试、多个协议上报告 p 值，需要在论文中说明多重比较校正方法。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "runwise_paired"

EXPERIMENT_ROOTS = {
    "baseline": PROJECT_ROOT / "outputs" / "loro_lodo",
    "inductive": PROJECT_ROOT / "outputs" / "runwise_inductive",
    "transductive": PROJECT_ROOT / "outputs" / "runwise_transductive",
}
SMOKE_TOKENS = ("_smoke", "_test")

CONTRASTS = {
    "inductive": "inductive_accuracy",
    "transductive": "transductive_accuracy",
}

PAIR_FIELDS = [
    "protocol",
    "subject",
    "group",
    "n_test",
    "baseline_accuracy",
    "inductive_accuracy",
    "transductive_accuracy",
    "delta_inductive_minus_baseline",
    "delta_transductive_minus_baseline",
]
SUMMARY_FIELDS = [
    "protocol",
    "level",
    "subject",
    "contrast",
    "n_pairs",
    "mean_baseline",
    "mean_normalized",
    "mean_diff",
    "median_diff",
    "ci_low",
    "ci_high",
    "wilcoxon_p",
    "n_better",
    "n_equal",
    "n_worse",
]


def _is_temp(path: Path) -> bool:
    """跳过 smoke / test 临时目录。"""
    return any(token in part for part in path.parts for token in SMOKE_TOKENS)


def _latest_run(root: Path) -> Path | None:
    """返回 root 下最新的、含逐折 CSV（folds_* 或 paired_*）的运行目录。"""
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and not _is_temp(path)
        and (any(path.rglob("folds_*.csv")) or any(path.rglob("paired_*.csv")))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _find(root: Path, name: str) -> Path | None:
    """在运行目录下递归查找文件，兼容单被试与多被试两种输出布局。"""
    if not root.exists():
        return None
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    """对逐组差值重采样，返回均值的 95% 百分位区间。"""
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_stats(diffs: np.ndarray, seed: int) -> dict:
    """一组配对差值的描述统计与检验。"""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    ci_low, ci_high = _bootstrap_ci(diffs, seed)
    try:
        p_value = float(wilcoxon(diffs).pvalue)
    except ValueError:  # 全部差值为 0 时无法检验
        p_value = None
    return {
        "n_pairs": int(len(diffs)),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": p_value,
        "n_better": int(np.sum(diffs > 0)),
        "n_equal": int(np.sum(diffs == 0)),
        "n_worse": int(np.sum(diffs < 0)),
    }


def load_protocol(protocol: str, roots: dict[str, Path]) -> pd.DataFrame:
    """按 ``(subject, group)`` 对齐三种口径，返回逐组长表。"""
    base_root = roots["baseline"]
    prefix = f"paired_{protocol}_"
    subjects = sorted(
        path.stem[len(prefix) :] for path in base_root.rglob(f"{prefix}*.csv")
    )
    if not subjects:
        print(f"[跳过] {base_root} 下找不到 {prefix}*.csv")
        return pd.DataFrame(columns=PAIR_FIELDS)

    frames: list[pd.DataFrame] = []
    for subject in subjects:
        base_path = _find(base_root, f"{prefix}{subject}.csv")
        if base_path is None:
            continue

        base = _read(base_path)[["group", "n_test", "loo_accuracy"]].rename(
            columns={"loo_accuracy": "baseline_accuracy"}
        )
        base["protocol"] = protocol
        base["subject"] = subject

        merged = base
        for key, column in CONTRASTS.items():
            path = _find(roots[key], f"folds_{protocol}_{subject}.csv")
            if path is None:
                print(f"[缺失] {key}: {roots[key]} 下没有 folds_{protocol}_{subject}.csv")
                merged[column] = np.nan
                continue
            other = _read(path)[["split", "n_test", "accuracy"]].rename(
                columns={"split": "group", "accuracy": column, "n_test": f"n_test_{key}"}
            )
            merged = merged.merge(other, on="group", how="outer", suffixes=("", f"_{key}"))
            # 两种口径的 n_test 必须一致，否则说明两次实验选的类或样本不同
            mismatch = merged[f"n_test_{key}"].notna() & (
                merged[f"n_test_{key}"] != merged["n_test"]
            )
            if bool(mismatch.any()):
                print(
                    f"[警告] {subject} {protocol} 的 {key} 有 "
                    f"{int(mismatch.sum())} 个组 n_test 与基线不一致"
                )
            merged = merged.drop(columns=[f"n_test_{key}"])

        merged["subject"] = subject
        merged["protocol"] = protocol
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=PAIR_FIELDS)

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["subject", "group"]).reset_index(drop=True)
    frame["protocol"] = protocol
    frame["delta_inductive_minus_baseline"] = (
        frame["inductive_accuracy"] - frame["baseline_accuracy"]
    )
    frame["delta_transductive_minus_baseline"] = (
        frame["transductive_accuracy"] - frame["baseline_accuracy"]
    )
    return frame


def summarize(frame: pd.DataFrame, protocol: str, seed: int) -> pd.DataFrame:
    """逐被试 + 跨被试的配对统计。"""
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_FIELDS)

    for subject, part in frame.groupby("subject", sort=True):
        for key, column in CONTRASTS.items():
            valid = part[[column, "baseline_accuracy"]].dropna()
            if valid.empty:
                continue
            stats = _paired_stats(valid[column].to_numpy() - valid["baseline_accuracy"].to_numpy(), seed)
            rows.append(
                {
                    "protocol": protocol,
                    "level": "subject",
                    "subject": subject,
                    "contrast": f"{key}_minus_baseline",
                    "mean_baseline": float(valid["baseline_accuracy"].mean()),
                    "mean_normalized": float(valid[column].mean()),
                    **stats,
                }
            )

    # 跨被试：每个被试先取自身均值，再对被试均值做配对检验（n=被试数）
    for key, column in CONTRASTS.items():
        subject_diffs = []
        for _, part in frame.groupby("subject", sort=True):
            valid = part[[column, "baseline_accuracy"]].dropna()
            if not valid.empty:
                subject_diffs.append((valid[column] - valid["baseline_accuracy"]).mean())
        if not subject_diffs:
            continue
        stats = _paired_stats(np.asarray(subject_diffs, dtype=float), seed)
        rows.append(
            {
                "protocol": protocol,
                "level": "across_subject",
                "subject": "all",
                "contrast": f"{key}_minus_baseline",
                "mean_baseline": np.nan,
                "mean_normalized": np.nan,
                **stats,
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_FIELDS)


def print_summary(protocol: str, summary: pd.DataFrame) -> None:
    header = (
        f"{'level':<15}{'subject':<9}{'contrast':<26}{'n':>4}"
        f"{'base':>8}{'norm':>8}{'diff':>8}{'95% CI':>20}{'p':>9}{'better/worse':>13}"
    )
    print(f"\n=== [{protocol}] 配对结果（差值 = 归一化后 − 无归一化）===")
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        base = "-" if pd.isna(row["mean_baseline"]) else f"{row['mean_baseline']:.4f}"
        norm = "-" if pd.isna(row["mean_normalized"]) else f"{row['mean_normalized']:.4f}"
        p_text = "-" if pd.isna(row["wilcoxon_p"]) else f"{row['wilcoxon_p']:.4g}"
        ci_text = f"[{row['ci_low']:+.3f},{row['ci_high']:+.3f}]"
        trades = f"{int(row['n_better'])}/{int(row['n_worse'])}"
        print(
            f"{row['level']:<15}{row['subject']:<9}{row['contrast']:<26}{row['n_pairs']:>4}"
            f"{base:>8}{norm:>8}{row['mean_diff']:>+8.4f}{ci_text:>20}{p_text:>9}"
            f"{trades:>13}"
        )


def resolve_roots(args) -> dict[str, Path]:
    """把命令行参数落到具体运行目录；未指定时取最新非临时运行目录。"""
    roots: dict[str, Path] = {}
    explicit = {
        "baseline": args.baseline_run,
        "inductive": args.inductive_run,
        "transductive": args.transductive_run,
    }
    for key, value in explicit.items():
        if value is not None:
            roots[key] = value
            continue
        latest = _latest_run(EXPERIMENT_ROOTS[key])
        if latest is None:
            raise SystemExit(
                f"找不到 {key} 的运行目录：{EXPERIMENT_ROOTS[key]} 下没有含逐折 CSV 的子目录，"
                "请用 --baseline-run / --inductive-run / --transductive-run 显式指定"
            )
        roots[key] = latest
        print(f"[自动] {key} = {latest.relative_to(PROJECT_ROOT)}")
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐 run 配对比较：run-wise 归一化（归纳/直推）vs 无归一化。"
    )
    parser.add_argument("--baseline-run", type=Path, default=None, help="loro_lodo 的运行目录")
    parser.add_argument("--inductive-run", type=Path, default=None, help="runwise_inductive 的运行目录")
    parser.add_argument("--transductive-run", type=Path, default=None, help="runwise_transductive 的运行目录")
    parser.add_argument(
        "--protocol",
        default="both",
        choices=("loro", "lodo", "both"),
        help="按 run 留一（loro）还是按天留一（lodo）配对",
    )
    parser.add_argument("--seed", type=int, default=42, help="bootstrap 与检验的随机种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    args = parser.parse_args()

    roots = resolve_roots(args)
    protocols = ["loro", "lodo"] if args.protocol == "both" else [args.protocol]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for protocol in protocols:
        frame = load_protocol(protocol, roots)
        if frame.empty:
            print(f"\n[{protocol}] 没有可配对的组，跳过")
            continue
        pairs_path = args.out_dir / f"paired_{protocol}.csv"
        frame[[column for column in PAIR_FIELDS if column in frame]].to_csv(
            pairs_path, index=False, encoding="utf-8-sig"
        )
        print(f"\n[Saved] {pairs_path}")

        summary = summarize(frame, protocol, args.seed)
        summary_path = args.out_dir / f"paired_{protocol}_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print_summary(protocol, summary)
        print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
