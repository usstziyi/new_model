"""汇总 baseline / run_identity / loro_lodo 的逐被试 CSV（可选包含 sweep）。

三个实验各自输出 ``<run>/<subject>/summary_<subject>.csv``（loro_lodo 每个被试
有两种模式两行）。本脚本把它们读成长表，并额外生成一张跨被试的均值/标准差表，
避免手工拼表。

用法::

    uv run outputs/collect_experiment_summary.py
    uv run outputs/collect_experiment_summary.py \
        --baseline-run outputs/baseline_lda/baseline_20261117_120000 \
        --run-identity-run outputs/run_identity/run_identity_20261117_130000 \
        --loro-lodo-run outputs/loro_lodo/loro_lodo_20261117_140000

不指定 ``*-run`` 时，自动选择对应输出目录下最新的、包含 summary CSV 的运行目录；
名字带 ``_smoke`` / ``_test`` 的临时目录会被跳过。
输出默认写到 ``outputs/experiment_summary/``。

``--include-sweep`` 时，sweep 行的 ``protocol`` 记为 ``8:2-sweep``
（baseline / run_identity）或 ``loro_lodo-sweep``（loro_lodo），
``evaluation`` 统一为 ``sweep``，具体模式记录在 ``mode`` 列。

注意：若同时报告多个被试或多个频段配置的 p 值，需要在论文中说明多重比较校正方法
（如 Bonferroni 或 FDR）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "experiment_summary"

EXPERIMENTS = {
    "baseline_lda": PROJECT_ROOT / "outputs" / "baseline_lda",
    "run_identity": PROJECT_ROOT / "outputs" / "run_identity",
    "loro_lodo": PROJECT_ROOT / "outputs" / "loro_lodo",
}

SMOKE_TOKENS = ("_smoke", "_test")

LONG_FIELDS = [
    "experiment",
    "evaluation",
    "mode",
    "protocol",
    "config",
    "subject",
    "eeg_task",
    "top_k",
    "band_mode",
    "model",
    "n_features",
    "n_samples",
    "n_groups",
    "val_ratio",
    "test_ratio",
    "reference_cv_folds",
    "chance",
    "accuracy",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "macro_f1",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_macro_f1_mean",
    "cv_accuracy_ci_low",
    "cv_accuracy_ci_high",
    "permutation_n",
    "permutation_p",
    "paired_mean_diff",
    "paired_ci_low",
    "paired_ci_high",
]


def _is_temp(path: Path) -> bool:
    """跳过 smoke / test 临时目录或文件。"""
    return any(token in part for part in path.parts for token in SMOKE_TOKENS)


def _latest_run(root: Path) -> Path | None:
    """返回 root 下最新的、包含 summary CSV 的运行目录。"""
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and not _is_temp(path) and any(path.rglob("summary_*.csv"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _summary_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("summary_*.csv"))
        if path.name != "summary_all.csv" and not _is_temp(path)
    ]


def _sweep_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("sweep_*.csv"))
        if path.name != "sweep_all.csv" and not _is_temp(path)
    ]


def _pick(frame: pd.DataFrame, *names: str) -> pd.Series:
    """按顺序取第一个存在的列，否则返回全 NA。"""
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([np.nan] * len(frame), index=frame.index)


def _protocol(raw: pd.DataFrame, experiment: str) -> pd.Series:
    """给每行标注协议，便于过滤掉旧口径结果。"""
    if experiment == "loro_lodo":
        return pd.Series(["loro_lodo"] * len(raw), index=raw.index)
    if "val_ratio" in raw.columns and "test_ratio" in raw.columns:
        val_ratio = pd.to_numeric(raw["val_ratio"], errors="coerce").fillna(-1)
        test_ratio = pd.to_numeric(raw["test_ratio"], errors="coerce").fillna(-1)
        return pd.Series(
            np.where((val_ratio == 0) & (test_ratio == 0.2), "8:2", "custom"),
            index=raw.index,
        )
    return pd.Series(["unknown"] * len(raw), index=raw.index)


def _empty_like(index: pd.Index, value=np.nan) -> pd.Series:
    return pd.Series([value] * len(index), index=index)


META_SOURCES = {
    "eeg_task": ("eeg_task",),
    "top_k": ("top_k",),
    "band_mode": ("band_mode",),
    "model": ("model",),
    "n_samples": ("n_samples",),
    "n_groups": ("n_runs", "n_groups"),
}


def _subject_meta(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """从被试级 summary 提取元数据，供没有记录这些列的 sweep CSV 回填。"""
    rows = []
    for raw in frames:
        if raw.empty or "subject" not in raw.columns:
            continue
        row = {"subject": raw["subject"].iloc[0]}
        for field, names in META_SOURCES.items():
            for name in names:
                if name in raw.columns:
                    value = raw[name].iloc[0]
                    if pd.notna(value):
                        row[field] = value
                        break
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("subject").set_index("subject")


def _from_raw_or_meta(
    raw: pd.DataFrame, meta: pd.DataFrame, field: str, *names: str
) -> pd.Series:
    """优先取 CSV 自身列，否则按被试从 summary 元数据回填。"""
    for name in names or (field,):
        if name in raw.columns:
            return raw[name]
    if not meta.empty and field in meta.columns:
        return raw["subject"].map(meta[field])
    return _empty_like(raw.index)


def _sweep_protocol(experiment: str) -> str:
    """sweep 与主实验同口径：baseline/run_identity 是 8:2 留出，loro_lodo 是跨被试。"""
    return "loro_lodo-sweep" if experiment == "loro_lodo" else "8:2-sweep"


def _harmonize_summary(raw: pd.DataFrame, experiment: str) -> pd.DataFrame:
    if "mode" in raw.columns:
        evaluation = raw["mode"]
    elif experiment == "baseline_lda":
        evaluation = _empty_like(raw.index, "random_split")
    else:
        evaluation = _empty_like(raw.index, "run_identity")

    return pd.DataFrame(
        {
            "experiment": raw["experiment"],
            "evaluation": evaluation,
            "mode": _pick(raw, "mode"),
            "protocol": _protocol(raw, experiment),
            "config": _empty_like(raw.index),
            "subject": raw.get("subject"),
            "eeg_task": raw.get("eeg_task"),
            "top_k": raw.get("top_k"),
            "band_mode": raw.get("band_mode"),
            "model": raw.get("model"),
            "n_features": raw.get("n_features"),
            "n_samples": raw.get("n_samples"),
            "n_groups": _pick(raw, "n_runs", "n_groups"),
            "val_ratio": _pick(raw, "val_ratio"),
            "test_ratio": _pick(raw, "test_ratio"),
            "reference_cv_folds": _pick(raw, "reference_cv_folds"),
            "chance": _pick(raw, "chance", "chance_accuracy"),
            "accuracy": _pick(raw, "test_accuracy", "pooled_accuracy"),
            "accuracy_ci_low": _pick(raw, "test_accuracy_ci_low", "pooled_ci_low"),
            "accuracy_ci_high": _pick(raw, "test_accuracy_ci_high", "pooled_ci_high"),
            "macro_f1": _pick(raw, "test_macro_f1", "pooled_macro_f1"),
            "cv_accuracy_mean": _pick(raw, "cv_accuracy_mean", "reference_cv_accuracy_mean"),
            "cv_accuracy_std": _pick(raw, "cv_accuracy_std", "reference_cv_accuracy_std"),
            "cv_macro_f1_mean": _pick(raw, "cv_macro_f1_mean", "reference_cv_macro_f1_mean"),
            "cv_accuracy_ci_low": _pick(raw, "cv_accuracy_ci_low"),
            "cv_accuracy_ci_high": _pick(raw, "cv_accuracy_ci_high"),
            "permutation_n": _pick(raw, "permutation_n"),
            "permutation_p": _pick(raw, "permutation_p"),
            "paired_mean_diff": _pick(raw, "paired_mean_diff"),
            "paired_ci_low": _pick(raw, "paired_ci_low"),
            "paired_ci_high": _pick(raw, "paired_ci_high"),
        }
    )[LONG_FIELDS]


def _harmonize_sweep(
    raw: pd.DataFrame, experiment: str, meta: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment": raw["experiment"],
            "evaluation": _empty_like(raw.index, "sweep"),
            "mode": _pick(raw, "mode"),
            "protocol": _empty_like(raw.index, _sweep_protocol(experiment)),
            "config": raw.get("config"),
            "subject": raw.get("subject"),
            "eeg_task": _from_raw_or_meta(raw, meta, "eeg_task"),
            "top_k": _from_raw_or_meta(raw, meta, "top_k"),
            "band_mode": _from_raw_or_meta(raw, meta, "band_mode"),
            "model": _from_raw_or_meta(raw, meta, "model"),
            "n_features": raw.get("n_features"),
            "n_samples": _from_raw_or_meta(raw, meta, "n_samples"),
            "n_groups": _from_raw_or_meta(raw, meta, "n_groups", "n_runs", "n_groups"),
            "val_ratio": _empty_like(raw.index),
            "test_ratio": _empty_like(raw.index),
            "reference_cv_folds": _empty_like(raw.index),
            "chance": _pick(raw, "chance"),
            "accuracy": _pick(raw, "test_accuracy", "loo_accuracy"),
            "accuracy_ci_low": _pick(raw, "test_accuracy_ci_low", "loo_ci_low"),
            "accuracy_ci_high": _pick(raw, "test_accuracy_ci_high", "loo_ci_high"),
            "macro_f1": _pick(raw, "test_macro_f1", "loo_macro_f1"),
            "cv_accuracy_mean": _pick(raw, "cv_accuracy_mean", "reference_cv_accuracy"),
            "cv_accuracy_std": _pick(raw, "cv_accuracy_std"),
            "cv_macro_f1_mean": _pick(raw, "cv_macro_f1_mean"),
            "cv_accuracy_ci_low": _pick(raw, "cv_accuracy_ci_low"),
            "cv_accuracy_ci_high": _pick(raw, "cv_accuracy_ci_high"),
            "permutation_n": _empty_like(raw.index),
            "permutation_p": _empty_like(raw.index),
            "paired_mean_diff": _pick(raw, "paired_mean_diff"),
            "paired_ci_low": _empty_like(raw.index),
            "paired_ci_high": _empty_like(raw.index),
        }
    )[LONG_FIELDS]


def _load_run(run_dir: Path, experiment: str, include_sweep: bool) -> pd.DataFrame:
    summary_paths = _summary_files(run_dir)
    sweep_paths = _sweep_files(run_dir) if include_sweep else []

    summary_frames = []
    for path in summary_paths:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw["experiment"] = experiment
        summary_frames.append(raw)

    meta = _subject_meta(summary_frames)
    frames = [_harmonize_summary(raw, experiment) for raw in summary_frames]
    for path in sweep_paths:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw["experiment"] = experiment
        frames.append(_harmonize_sweep(raw, experiment, meta))

    if not frames:
        print(f"[Skip] {experiment}: {run_dir} 下没有可汇总的 CSV")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    return (
        long.groupby(
            ["experiment", "evaluation", "protocol", "config", "mode"], dropna=False
        )
        .agg(
            n_subjects=("subject", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            cv_accuracy_mean=("cv_accuracy_mean", "mean"),
            cv_accuracy_between_subject_std=("cv_accuracy_mean", "std"),
            cv_within_subject_std_mean=("cv_accuracy_std", "mean"),
            cv_macro_f1_mean=("cv_macro_f1_mean", "mean"),
            paired_mean_diff=("paired_mean_diff", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 baseline / run_identity / loro_lodo 的逐被试 CSV（可选包含 sweep）。")
    parser.add_argument("--baseline-run", type=Path, help="baseline_lda 运行目录；默认取最新")
    parser.add_argument("--run-identity-run", type=Path, help="run_identity 运行目录；默认取最新")
    parser.add_argument("--loro-lodo-run", type=Path, help="loro_lodo 运行目录；默认取最新")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--include-sweep", action="store_true", help="同时汇总 sweep_<subject>.csv")
    args = parser.parse_args()

    run_dirs = {
        "baseline_lda": args.baseline_run or _latest_run(EXPERIMENTS["baseline_lda"]),
        "run_identity": args.run_identity_run or _latest_run(EXPERIMENTS["run_identity"]),
        "loro_lodo": args.loro_lodo_run or _latest_run(EXPERIMENTS["loro_lodo"]),
    }

    frames = []
    for experiment, run_dir in run_dirs.items():
        if run_dir is None:
            print(f"[Skip] {experiment}: 未找到运行目录")
            continue
        print(f"[Load] {experiment}: {run_dir}")
        frame = _load_run(run_dir, experiment, args.include_sweep)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit(
            "没有找到任何可汇总的 CSV；请先按当前 8:2 口径重跑 baseline / "
            "run_identity / loro_lodo。"
        )

    long = pd.concat(frames, ignore_index=True)

    aggregate = _aggregate(long)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.out_dir / "summary_all.csv"
    aggregate_path = args.out_dir / "summary_by_experiment.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    display = aggregate.copy()
    for column in display.columns:
        if display[column].dtype.kind == "f":
            display[column] = display[column].round(4)
    print("\n=== 跨被试汇总 ===")
    print(display.to_string(index=False))
    print(f"\n[Saved] {long_path}")
    print(f"[Saved] {aggregate_path}")


if __name__ == "__main__":
    main()
