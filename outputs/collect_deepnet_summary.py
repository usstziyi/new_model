"""汇总线性基线与两个深度基线（EEGNet / 频域 CNN）的跨协议结果。

数据源：

- 线性 LDA：``outputs/baseline_lda/main_8_2``（cv5）、``outputs/loro_lodo/main_8_2``
  （LORO / LODO）、``outputs/run_identity/main_8_2``（run 身份）
- EEGNet：``outputs/deepnet/eegnet/main_eegnet``、``outputs/deepnet/eegnet/runid_eegnet``
- 频域 CNN：``outputs/deepnet/freqcnn/main_freqcnn``、``outputs/deepnet/freqcnn/runid_freqcnn``

输出（默认写到 ``paper/tables/``）：

- ``table_deepnet_subject.csv``：逐被试长表，含三个协议、run 身份、虚高与优势保留率；
- ``table_deepnet_main.csv`` / ``.md``：论文用主表（跨被试均值 ± 标准差）；
- ``table_deepnet_contrasts.csv`` / ``.md``：关键对比的配对检验（n=5）。

口径说明：

- cv5 一律取逐折准确率均值（``cv_accuracy_mean`` / ``fold_mean_accuracy``），
  与线性基线的 ``cv_accuracy_mean`` 对齐；
- LORO / LODO 一律取池化准确率（``pooled_accuracy`` / ``accuracy``）；
- run 身份取随机 5 折下的交叉验证准确率；
- ``inflation = cv5 - loro``；``retention = (loro - 0.1) / (cv5 - 0.1)``，
  即超出 chance 的优势中被保留下来的比例，负值表示已低于 chance。

注意：n=5 时 Wilcoxon 符号秩检验的最小可能 p 值为 0.0625，因此不显著不等于无效应；
表中同时报告均值差、95% CI 与配对 Cohen's dz。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_OUT_DIR = PROJECT_ROOT / "paper" / "tables"
CHANCE = 0.10
RUN_CHANCE = 1 / 45

MODEL_LABELS = {
    "linear": "线性 LDA",
    "eegnet": "EEGNet",
    "freqcnn": "频域 CNN",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_linear() -> pd.DataFrame:
    """返回逐被试的 cv5 / loro / lodo / runid 四项准确率。"""
    cv = _read(OUTPUTS / "baseline_lda/main_8_2/summary_all.csv")
    cv = cv.set_index("subject")["cv_accuracy_mean"].rename("cv5")

    loo = _read(OUTPUTS / "loro_lodo/main_8_2/summary_all.csv")
    loro = (
        loo[loo["mode"] == "loro"].set_index("subject")["pooled_accuracy"].rename("loro")
    )
    lodo = (
        loo[loo["mode"] == "lodo"].set_index("subject")["pooled_accuracy"].rename("lodo")
    )

    rid = _read(OUTPUTS / "run_identity/main_8_2/summary_all.csv")
    runid = rid.set_index("subject")["cv_accuracy_mean"].rename("runid")

    frame = pd.concat([cv, loro, lodo, runid], axis=1)
    frame["model"] = "linear"
    frame["n_params"] = 610 * 10
    return frame.reset_index().rename(columns={"index": "subject"})


def load_deep(experiment: str, main_run: str, runid_run: str, n_params: int) -> pd.DataFrame:
    """读取一个深度实验的 cv5 / LORO / LODO / run 身份。"""
    main = _read(OUTPUTS / "deepnet" / experiment / main_run / "summary_all.csv")
    main = main.assign(
        cv5=np.where(
            main["protocol"] == "cv5", main["fold_mean_accuracy"], np.nan
        ),
        loro=np.where(main["protocol"] == "loro", main["accuracy"], np.nan),
        lodo=np.where(main["protocol"] == "lodo", main["accuracy"], np.nan),
    )
    grouped = main.groupby("subject")[["cv5", "loro", "lodo"]].first()

    rid = _read(OUTPUTS / "deepnet" / experiment / runid_run / "summary_all.csv")
    runid = rid.set_index("subject")["accuracy"].rename("runid")

    frame = grouped.join(runid)
    frame["model"] = experiment
    frame["n_params"] = n_params
    return frame.reset_index()


def build_subject_table() -> pd.DataFrame:
    frames = [
        load_linear(),
        load_deep("eegnet", "main_eegnet", "runid_eegnet", 13474),
        load_deep("freqcnn", "main_freqcnn", "runid_freqcnn", 129034),
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["inflation_loro"] = frame["cv5"] - frame["loro"]
    frame["inflation_lodo"] = frame["cv5"] - frame["lodo"]
    frame["retention_loro"] = (frame["loro"] - CHANCE) / (frame["cv5"] - CHANCE)
    frame["retention_lodo"] = (frame["lodo"] - CHANCE) / (frame["cv5"] - CHANCE)
    frame["model_label"] = frame["model"].map(MODEL_LABELS)
    return frame[["model", "model_label", "subject", "n_params", "cv5", "loro", "lodo", "runid",
                  "inflation_loro", "inflation_lodo", "retention_loro", "retention_lodo"]]


def build_main_table(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("linear", "eegnet", "freqcnn"):
        part = subject[subject["model"] == model]
        row = {
            "model": MODEL_LABELS[model],
            "n_params": int(part["n_params"].iloc[0]),
            "n_subjects": len(part),
        }
        for column, label in (
            ("cv5", "random_cv5"),
            ("loro", "loro"),
            ("lodo", "lodo"),
            ("runid", "run_identity"),
        ):
            row[f"{label}_mean"] = part[column].mean()
            row[f"{label}_sd"] = part[column].std(ddof=1)
        row["inflation_loro"] = part["inflation_loro"].mean()
        row["retention_loro"] = part["retention_loro"].mean()
        row["above_chance_loro"] = part["loro"].mean() - CHANCE
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        lo = hi = diff.mean()
        dz = np.nan
    else:
        se = sd / np.sqrt(len(diff))
        lo, hi = t_dist.interval(0.95, len(diff) - 1, diff.mean(), se)
        dz = diff.mean() / sd
    return {
        "mean_diff": diff.mean(),
        "ci_low": lo,
        "ci_high": hi,
        "t_p": ttest_rel(a, b).pvalue,
        "wilcoxon_p": wilcoxon(a, b).pvalue,
        "cohens_dz": dz,
        "n_better": int((diff > 0).sum()),
        "n_worse": int((diff < 0).sum()),
    }


def _one_sample(a: np.ndarray, ref: float) -> dict:
    a = np.asarray(a, dtype=float)
    diff = a - ref
    return {
        "mean_diff": diff.mean(),
        "t_p": ttest_1samp(a, ref).pvalue,
        "wilcoxon_p": wilcoxon(diff).pvalue,
        "cohens_dz": diff.mean() / diff.std(ddof=1),
    }


def build_contrasts(subject: pd.DataFrame) -> pd.DataFrame:
    pivot = subject.pivot(index="subject", columns="model", values=["cv5", "loro", "lodo", "runid"])
    rows = []

    def add(name, group, a, b=None, ref=None):
        stats = _paired(a, b) if b is not None else _one_sample(a, ref)
        row = {"contrast": name, "group": group}
        row.update(stats)
        if ref is not None:
            row["reference"] = ref
            row["cohens_dz"] = (np.asarray(a) - ref).mean() / np.asarray(a).std(ddof=1)
        rows.append(row)

    linear = pivot.xs("linear", axis=1, level=1)
    eeg = pivot.xs("eegnet", axis=1, level=1)
    fcnn = pivot.xs("freqcnn", axis=1, level=1)

    # 与 chance 比较
    for label, frame in (("线性 LDA", linear), ("EEGNet", eeg), ("频域 CNN", fcnn)):
        for protocol in ("cv5", "loro", "lodo"):
            add(f"{label}：{protocol} vs chance", "vs_chance", frame[protocol], ref=CHANCE)

    # 与线性基线比较（被试级）
    for label, frame in (("EEGNet", eeg), ("频域 CNN", fcnn)):
        for protocol in ("cv5", "loro", "lodo"):
            add(f"{label} − 线性 LDA（{protocol}）", "vs_linear",
                frame[protocol], linear[protocol])

    # 深度模型之间
    for protocol in ("cv5", "loro", "lodo"):
        add(f"EEGNet − 频域 CNN（{protocol}）", "deep_vs_deep", eeg[protocol], fcnn[protocol])

    # 随机划分相对留出的虚高
    for label, frame in (("线性 LDA", linear), ("EEGNet", eeg), ("频域 CNN", fcnn)):
        add(f"{label}：cv5 − LORO", "inflation", frame["cv5"], frame["loro"])
    return pd.DataFrame(rows)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}"


def write_markdown(subject: pd.DataFrame, main: pd.DataFrame, contrasts: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 深度基线主结果（5 名被试，10 类句类，chance 10%）",
        "",
        "## 表 A 三模型 × 四协议",
        "",
        "| 模型 | 参数 | 随机 5 折 | 留一 run（LORO） | 留一采集日（LODO） | run 身份 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"| {row['model']} | {row['n_params']:,} | "
            f"{_pct(row['random_cv5_mean'])} ± {_pct(row['random_cv5_sd'])} | "
            f"{_pct(row['loro_mean'])} ± {_pct(row['loro_sd'])} | "
            f"{_pct(row['lodo_mean'])} ± {_pct(row['lodo_sd'])} | "
            f"{_pct(row['run_identity_mean'])} ± {_pct(row['run_identity_sd'])} |"
        )
    lines += [
        f"| chance | — | 10.0 | 10.0 | 10.0 | {_pct(RUN_CHANCE)} |",
        "",
        "注：cv5 为逐折准确率均值；LORO / LODO 为池化准确率；run 身份为随机 5 折交叉验证"
        "（45 类，sub-05 为 44 类）。数值为 5 名被试的均值 ± 标准差（单位 %）。",
        "",
        "## 表 B 随机划分虚高与优势保留",
        "",
        "| 模型 | cv5 − LORO（pp） | 超出 chance 的优势保留率 | LORO 超出 chance（pp） |",
        "|---|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"| {row['model']} | {row['inflation_loro'] * 100:.2f} | "
            f"{row['retention_loro'] * 100:.1f}% | {row['above_chance_loro'] * 100:.2f} |"
        )
    lines += [
        "",
        "注：优势保留率 = (LORO − 10%) / (cv5 − 10%)；负值表示 LORO 已低于 chance。",
        "",
        "## 表 C 关键对比（n = 5 名被试）",
        "",
        "| 对比 | 类别 | 平均差（pp） | 95% CI（pp） | 配对 t | Wilcoxon | dz | 变好/变差 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in contrasts.iterrows():
        ci = (
            f"[{row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f}]"
            if "ci_low" in row and pd.notna(row.get("ci_low"))
            else "—"
        )
        dz = f"{row['cohens_dz']:+.2f}" if pd.notna(row.get("cohens_dz")) else "—"
        better = (
            f"{int(row['n_better'])}/{int(row['n_worse'])}"
            if "n_better" in row and pd.notna(row.get("n_better"))
            else "—"
        )
        lines.append(
            f"| {row['contrast']} | {row['group']} | {row['mean_diff'] * 100:+.2f} | {ci} | "
            f"{row['t_p']:.4f} | {row['wilcoxon_p']:.4f} | {dz} | {better} |"
        )
    lines += [
        "",
        "注：n = 5 时 Wilcoxon 符号秩检验的最小可能 p 值为 0.0625；"
        "因此“全部同向但不显著”与“无效应”不能混为一谈。多重比较未校正，"
        "论文中需说明 Bonferroni 或 FDR 口径。",
        "",
    ]
    (out_dir / "table_deepnet_main.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总线性基线与两个深度基线的跨协议结果。")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    args = parser.parse_args()

    subject = build_subject_table()
    main_table = build_main_table(subject)
    contrasts = build_contrasts(subject)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject.to_csv(args.out_dir / "table_deepnet_subject.csv", index=False, encoding="utf-8-sig")
    main_table.to_csv(args.out_dir / "table_deepnet_main.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(args.out_dir / "table_deepnet_contrasts.csv", index=False, encoding="utf-8-sig")
    write_markdown(subject, main_table, contrasts, args.out_dir)

    with pd.option_context("display.width", 220):
        print("=== 表 A 三模型 × 四协议（%）===")
        for _, row in main_table.iterrows():
            print(
                f"{row['model']:<10}{row['random_cv5_mean']*100:6.2f}±{row['random_cv5_sd']*100:4.2f}"
                f"{row['loro_mean']*100:9.2f}±{row['loro_sd']*100:4.2f}"
                f"{row['lodo_mean']*100:9.2f}±{row['lodo_sd']*100:4.2f}"
                f"{row['run_identity_mean']*100:9.2f}±{row['run_identity_sd']*100:4.2f}"
            )
        print("\n=== 表 C 关键对比 ===")
        print(
            contrasts[["contrast", "group", "mean_diff", "t_p", "wilcoxon_p", "cohens_dz"]]
            .assign(mean_diff=lambda d: d.mean_diff * 100)
            .round(4)
            .to_string(index=False)
        )
    print(f"\n[Saved] {args.out_dir}/table_deepnet_{{subject,main,contrasts}}.csv + table_deepnet_main.md")


if __name__ == "__main__":
    main()
