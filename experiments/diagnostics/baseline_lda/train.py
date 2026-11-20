"""训练并评估线性基线：log 频带功率 + 收缩 LDA / 逻辑回归 / RBF SVM。

深度模型的验证/测试指标必须和这里的数字比较，过不了这条线就不该继续加大模型。
数据直接由 ``datasets/preprocessed/<subject>_<task>.csv`` 选 top-k 句类并从原始
run pkl 加载（见 ``dataset.py``），不经过 ``top10_imagine`` / ``top10_reading``
中间产物。

评估内容：

1. **同划分协议**：沿用与 ``plan_top10_random`` / PMCNN 一致的类别分层随机
   划分思路（同 ``split_seed``；本实验默认 8:2、train:test = 4:1、不设 val，
   数据由本实验独立从 CSV + 原始 pkl 加载），在 test 上给出准确率与 macro-F1，
   可与 PMCNN 的 ``history.csv`` 对比；
2. **5 折交叉验证**：全量样本上的稳定估计（均值 ± 标准差，附 run 级 bootstrap 95% CI）；
3. **置换零分布**：全局打乱标签后重复 5 折，给出随机水平与经验 p 值；
4. **频段扫描**（``--sweep-bands``）：all / 单频段 / 留一频段共 11 种配置，Welch 只算一次；扫描模式不做置换。

注意：若同时报告多个被试或多个频段配置的 p 值，需要在论文中说明多重比较校正方法
（如 Bonferroni 或 FDR），不要直接把未校正 p 当作独立证据。

特征与分类器见 ``model.py``：逐通道 5 个频段的 log 功率（默认相对功率），


相对功率对逐试次 z-score 不敏感，因此这里关闭 ``normalize``，直接在 µV 上做 Welch。
分类器由 ``--model`` 选择，默认收缩 LDA。

用法::


    uv run experiments/diagnostics/baseline_lda/train.py --sweep-bands --run-name sweep_rel_lda


    uv run experiments/diagnostics/baseline_lda/train.py
    uv run experiments/diagnostics/baseline_lda/train.py --subject sub-01 sub-02
    uv run experiments/diagnostics/baseline_lda/train.py --task reading      # 句朗读任务
    uv run experiments/diagnostics/baseline_lda/train.py --top-k 10
    uv run experiments/diagnostics/baseline_lda/train.py --model svm      # RBF 核 SVM
    uv run experiments/diagnostics/baseline_lda/train.py --model logreg
    uv run experiments/diagnostics/baseline_lda/train.py --use-64-channels
    uv run experiments/diagnostics/baseline_lda/train.py --permutations 100 --folds 5
    uv run experiments/diagnostics/baseline_lda/train.py --bands-mode abs      # 改用绝对 log 功率
    uv run experiments/diagnostics/baseline_lda/train.py --sweep-bands         # 单频段 + 留一频段扫描
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from dataset import CSV_ROOT, PKL_ROOT, TopClassEEGDataset, split_indices
from model import BANDS, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline_lda"

REPORT_FIELDS = ["subject", "label", "precision", "recall", "f1", "support"]
SUMMARY_FIELDS = [
    "subject",
    "eeg_task",
    "top_k",
    "band_mode",
    "model",
    "n_features",
    "n_samples",
    "n_runs",
    "val_ratio",
    "test_ratio",
    "split_seed",
    "chance_accuracy",
    "val_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_accuracy_ci_low",
    "test_accuracy_ci_high",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_accuracy_ci_low",
    "cv_accuracy_ci_high",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "permutation_n",
    "permutation_null_mean",
    "permutation_null_p95",
    "permutation_z",
    "permutation_p",
]
SWEEP_FIELDS = [
    "subject",
    "config",
    "n_features",
    "val_accuracy",
    "test_accuracy",
    "test_accuracy_ci_low",
    "test_accuracy_ci_high",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_accuracy_ci_low",
    "cv_accuracy_ci_high",
    "cv_macro_f1_mean",
]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """把字典列表写成 UTF-8-SIG CSV（Excel 可直接打开）。"""
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_csvs(paths: list[Path], out_path: Path) -> None:
    """把多个同结构 CSV 拼成一个（用于跨被试汇总）。"""
    rows: list[dict] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            rows.extend(reader)
    if not rows:
        return
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_classification_csv(path: Path, subject: str, report: dict) -> None:
    """把 ``classification_report(output_dict=True)`` 写成逐类 CSV。"""
    total_support = sum(
        metrics["support"]
        for label, metrics in report.items()
        if isinstance(metrics, dict) and label not in ("macro avg", "weighted avg", "samples avg")
    )
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {
                "subject": subject,
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
        elif label == "accuracy":  # output_dict 模式下 accuracy 是标量
            row = {
                "subject": subject,
                "label": label,
                "precision": metrics,
                "recall": metrics,
                "f1": metrics,
                "support": total_support,
            }
        else:
            continue
        rows.append(row)
    write_rows(path, rows, REPORT_FIELDS)


def bootstrap_accuracy_ci(
    y_true,
    y_pred,
    groups,
    seed: int,
    n_boot: int = 5000,
) -> tuple[float, float]:
    """按 run 重采样的准确率 95% CI（cluster bootstrap）。

    每次有放回地抽取 run，再把这些 run 的全部试次拼起来算准确率；这样保留了
    run 内的相关性，比按试次独立重采样更保守。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([index_by_group[j] for j in pick])
        accs[i] = np.mean(y_true[index] == y_pred[index])
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def cross_validate(
    X,
    y,
    model_name: str,
    folds: int,
    seed: int,
    return_oof: bool = False,
):
    """返回每折的 (accuracy, macro_f1)；``return_oof=True`` 时额外返回 OOF 预测。"""
    accs, f1s = [], []
    oof = np.empty(len(y), dtype=np.asarray(y).dtype) if return_oof else None
    # 用于构造交叉验证的划分器，自身不会执行数据划分
    # 与普通 KFold 不同，它会分别针对每个类别按比例切分，保证每折中各类别的比例与全集一致
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    # 真正划分
    for train_idx, test_idx in splitter.split(X, y):
        model = make_model(model_name)
        # 训练模型
        model.fit(X[train_idx], y[train_idx])
        # 预测测试集
        pred = model.predict(X[test_idx])
        if oof is not None:
            oof[test_idx] = pred
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    if oof is not None:
        return np.asarray(accs), np.asarray(f1s), oof
    return np.asarray(accs), np.asarray(f1s)


def permutation_null(X, y, model_name: str, folds: int, seed: int, n_perm: int) -> np.ndarray:
    """全局打乱标签后的 CV 准确率零分布。"""
    rng = np.random.default_rng(seed)
    # null 来自统计学的 null hypothesis（原假设或零假设）
    null = np.empty(n_perm)
    step = 10  # 每 10 次置换打印一行进度
    for i in range(n_perm):
        # 置换操作：全局打乱标签
        shuffled = rng.permutation(y)
        accs, _ = cross_validate(X, shuffled, model_name, folds, seed)
        null[i] = accs.mean()
        if (i + 1) % step == 0 or i == n_perm - 1:
            print(
                f"[置换] {i + 1}/{n_perm} ({(i + 1) / n_perm:.0%}) "
                f"本次 acc={null[i]:.4f}",
                flush=True,
            )

    return null


def holdout_report(X, y, train_idx, eval_sets, class_names, model_name, groups, seed):
    """在固定划分上训练一次，返回各评估集的指标（含 run 级 bootstrap CI）。"""
    model = make_model(model_name)
    model.fit(X[train_idx], y[train_idx])
    out = {}
    for name, idx in eval_sets.items():
        pred = model.predict(X[idx])
        ci_low, ci_high = bootstrap_accuracy_ci(y[idx], pred, groups[idx], seed)
        out[name] = {
            "accuracy": float(accuracy_score(y[idx], pred)),
            "accuracy_ci_low": ci_low,
            "accuracy_ci_high": ci_high,
            "macro_f1": float(f1_score(y[idx], pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[idx], pred, average="weighted", zero_division=0)),
            "n_samples": int(len(idx)),
            "confusion_matrix": confusion_matrix(
                y[idx], pred, labels=list(range(len(class_names)))
            ).tolist(),
            "per_class_recall": [
                float(np.mean(pred[y[idx] == c] == c)) if np.any(y[idx] == c) else None
                for c in range(len(class_names))
            ],
        }
    return model, out


def parse_args():
    parser = argparse.ArgumentParser(description="log 频带功率 + 线性分类器基线（LDA / logreg / SVM）。")
    parser.add_argument(
        "--subject", nargs="+", default=["sub-01"], help="被试编号，可指定多个，如 --subject sub-01 sub-02"
    )
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"), help="读取哪一组 CSV / run pkl")
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--sweep-bands", action="store_true", help="逐个频段扫描（all / 单频段 / 留一频段），Welch 只算一次；扫描模式不做置换")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器：lda 收缩 LDA（主基线）、logreg 逻辑回归或 svm RBF 核 SVM")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--permutations", type=int, default=2000, help="置换检验次数，0 表示跳过")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="验证集比例，默认 0（8:2，不设 val）")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="测试集比例（默认 0.2，对应 8:2）")
    parser.add_argument("--split-seed", type=int, default=42, help="划分种子（与 train.py 一致）")
    parser.add_argument("--seed", type=int, default=42, help="CV 与置换的随机种子")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    return parser.parse_args()


def band_columns(only, exclude, channels_per_band: int) -> tuple[np.ndarray, list[str]]:
    """返回按 ``BANDS`` 顺序排列的列索引，以及实际保留的频段名。"""
    bands = list(BANDS)
    kept = [
        name
        for name in bands
        if (not only or name in only) and name not in (exclude or [])
    ]
    if not kept:
        raise SystemExit("频段选择结果为空")

    index = []
    for name in kept:
        start = bands.index(name) * channels_per_band
        index.append(np.arange(start, start + channels_per_band))
    return np.concatenate(index), kept


def evaluate(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
    n_perm: int,
) -> tuple:
    """一次完整评估：holdout + 交叉验证 + 置换零分布。"""
    eval_sets = {"test": test_idx}
    if len(val_idx) > 0:
        eval_sets["val"] = val_idx
    model, holdout = holdout_report(
        feats, y, train_idx, eval_sets, class_names, model_name, groups, seed
    )
    accs, f1s, oof = cross_validate(feats, y, model_name, folds, seed, return_oof=True)
    cv_ci_low, cv_ci_high = bootstrap_accuracy_ci(y, oof, groups, seed)
    permutation = None
    if n_perm > 0:
        null = permutation_null(feats, y, model_name, folds, seed, n_perm)
        observed = float(accs.mean())
        permutation = {
            "n": n_perm,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "observed": observed,
            "z_score": float((observed - null.mean()) / (null.std() + 1e-12)),
            "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "null_samples": null.tolist(),
        }
    return model, {
        "n_features": int(feats.shape[1]),
        "holdout": holdout,
        "cv": {
            model_name: {
                "accuracy": accs.tolist(),
                "macro_f1": f1s.tolist(),
                "accuracy_ci_low": cv_ci_low,
                "accuracy_ci_high": cv_ci_high,
            }
        },
        "permutation": permutation,
    }


def sweep_configs(channels_per_band: int) -> list[tuple[str, np.ndarray]]:
    """all + 5 个单频段 + 5 个留一频段，共 11 种列选择。"""
    bands = list(BANDS)
    configs = [("all", np.arange(len(bands) * channels_per_band))]
    configs += [(f"only:{name}", band_columns([name], None, channels_per_band)[0]) for name in bands]
    configs += [(f"drop:{name}", band_columns(None, [name], channels_per_band)[0]) for name in bands]
    return configs


def print_sweep_table(rows: list[dict]) -> None:
    """打印频段扫描的汇总表。"""
    header = (
        f"{'config':<12}{'n_feat':>7}{'val':>8}{'test':>8}{'test_95CI':>17}"
        f"{'cv_acc':>9}{'std':>7}{'macroF1':>9}"
    )
    print("\n=== 频段扫描汇总 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        val_text = f"{row['val_accuracy']:.4f}" if row["val_accuracy"] is not None else "-"
        ci_text = f"[{row['test_accuracy_ci_low']:.3f},{row['test_accuracy_ci_high']:.3f}]"
        print(
            f"{row['config']:<12}{row['n_features']:>7}{val_text:>8}"
            f"{row['test_accuracy']:>8.4f}{ci_text:>17}{row['cv_accuracy_mean']:>9.4f}"
            f"{row['cv_accuracy_std']:>7.4f}{row['cv_macro_f1_mean']:>9.4f}"
        )


def run_sweep(
    feats, y, groups, class_names, train_idx, val_idx, test_idx, args, out_dir: Path, n_runs: int
) -> None:
    """逐个频段评估，并把汇总写入 ``sweep.json``（扫描模式不做置换）。"""
    rows = []
    for label, columns in sweep_configs(feats.shape[1] // len(BANDS)):
        print(f"\n=== [{label}] {len(columns)} 维 ===", flush=True)
        _, result = evaluate(
            feats[:, columns],
            y,
            groups,
            class_names,
            train_idx,
            val_idx,
            test_idx,
            args.model,
            args.folds,
            args.seed,
            0,  # 扫描模式强制关闭置换，避免 11 × B 倍成本
        )
        holdout = result["holdout"]
        accs = np.asarray(result["cv"][args.model]["accuracy"])
        f1s = np.asarray(result["cv"][args.model]["macro_f1"])
        cv_stats = result["cv"][args.model]
        rows.append(
            {
                "subject": args.subject,
                "config": label,
                "n_features": result["n_features"],
                "val_accuracy": holdout.get("val", {}).get("accuracy"),
                "test_accuracy": holdout["test"]["accuracy"],
                "test_accuracy_ci_low": holdout["test"]["accuracy_ci_low"],
                "test_accuracy_ci_high": holdout["test"]["accuracy_ci_high"],
                "cv_accuracy_mean": float(accs.mean()),
                "cv_accuracy_std": float(accs.std()),
                "cv_accuracy_ci_low": cv_stats["accuracy_ci_low"],
                "cv_accuracy_ci_high": cv_stats["accuracy_ci_high"],
                "cv_macro_f1_mean": float(f1s.mean()),
                "detail": result,
            }
        )
        val_text = (
            f"{rows[-1]['val_accuracy']:.4f}" if rows[-1]["val_accuracy"] is not None else "-"
        )
        print(
            f"  val={val_text} test={rows[-1]['test_accuracy']:.4f} "
            f"cv={rows[-1]['cv_accuracy_mean']:.4f}±{rows[-1]['cv_accuracy_std']:.4f}",
            flush=True,
        )

    print_sweep_table(rows)
    write_rows(
        out_dir / f"sweep_{args.subject}.csv",
        [{key: row[key] for key in SWEEP_FIELDS} for row in rows],
        SWEEP_FIELDS,
    )
    payload = {
        "subject": args.subject,
        "task": "baseline_lda_classification_sweep",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "classes": class_names,
        "n_samples": int(len(y)),
        "n_runs": n_runs,
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": float(args.target_sfreq),
        "use_64_channels": args.use_64_channels,
        "split": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.split_seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "sweep": rows,
    }
    sweep_path = out_dir / "sweep.json"
    with sweep_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {sweep_path}")


def run_subject(args, out_dir: Path) -> list[dict]:
    """完成一个被试的评估并落盘，返回 summary 行。"""
    dataset = TopClassEEGDataset(
        subject=args.subject,
        task=args.task,
        top_k=args.top_k,
        csv_root=args.csv_root,
        pkl_root=args.pkl_root,
        target_sfreq=args.target_sfreq,
        normalize=False,  # 相对功率尺度不变，无需逐试次 z-score
        use_64_channels=args.use_64_channels,
    )
    y = dataset.labels
    groups = np.asarray([entry["run"] for entry in dataset.meta], dtype=np.int64)
    class_names = list(dataset.class_names)
    num_classes = len(class_names)
    sfreq = float(args.target_sfreq)

    min_class = int(np.bincount(y).min())
    if args.folds > min_class:
        print(f"[Warn] 最小的句类只有 {min_class} 条，折数从 {args.folds} 降到 {min_class}")
        args.folds = min_class

    # feats: (N, 610)
    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    # ---------- 1. 与 train.py 相同的 8:2 划分（默认不设 val） ----------
    train_idx, val_idx, test_idx = split_indices(
        y, args.val_ratio, args.test_ratio, args.split_seed
    )
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if args.sweep_bands:
        run_sweep(
            feats, y, groups, class_names, train_idx, val_idx, test_idx, args, out_dir,
            len(dataset.run_ids),
        )
        return []

    model, result = evaluate(
        feats, y, groups, class_names, train_idx, val_idx, test_idx,
        args.model, args.folds, args.seed, args.permutations,
    )
    holdout = result["holdout"]
    for name in ("val", "test"):
        if name not in holdout:
            continue
        m = holdout[name]
        print(
            f"  [{args.model}] {name:<4} acc={m['accuracy']:.4f} "
            f"95% CI [{m['accuracy_ci_low']:.4f}, {m['accuracy_ci_high']:.4f}] "
            f"macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f} (n={m['n_samples']})"
        )
    print(f"\n测试集分类报告（{args.model}）:")
    pred_test = model.predict(feats[test_idx])
    report_dict = classification_report(
        y[test_idx],
        pred_test,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            y[test_idx],
            pred_test,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )
    report_path = out_dir / f"classification_report_{args.subject}.csv"
    save_classification_csv(report_path, args.subject, report_dict)
    print(f"[Saved] {report_path}")

    # ---------- 2. 5 折交叉验证 ----------
    print(f"=== {args.folds} 折交叉验证（全量 {len(y)} 条）===")
    accs = np.asarray(result["cv"][args.model]["accuracy"])
    f1s = np.asarray(result["cv"][args.model]["macro_f1"])
    cv_stats = result["cv"][args.model]
    for i, (acc, f1) in enumerate(zip(accs, f1s), start=1):
        print(f"  fold {i}  acc={acc:.4f}  macro_f1={f1:.4f}")
    print(
        f"  {args.model:<7} acc={accs.mean():.4f} ± {accs.std():.4f}   "
        f"95% CI [{cv_stats['accuracy_ci_low']:.4f}, {cv_stats['accuracy_ci_high']:.4f}]   "
        f"macro_f1={f1s.mean():.4f} ± {f1s.std():.4f}"
    )
    print(f"  {'chance':<7} acc={1 / num_classes:.4f}")

    # ---------- 3. 置换零分布 ----------
    permutation = result["permutation"]
    if permutation is not None:
        print(
            f"  null acc={permutation['null_mean']:.4f} ± {permutation['null_std']:.4f} "
            f"p95={permutation['null_p95']:.4f}   observed={permutation['observed']:.4f}   "
            f"z={permutation['z_score']:.1f}   p={permutation['p_value']:.4f}"
        )

    # ---------- 输出 ----------
    payload = {
        "subject": args.subject,
        "task": "baseline_lda_classification",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "classes": class_names,
        "n_samples": int(len(y)),
        "n_runs": len(dataset.run_ids),
        "n_features": result["n_features"],
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": sfreq,
        "use_64_channels": args.use_64_channels,
        "split": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.split_seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "holdout": holdout,
        "cv": result["cv"],
        "permutation": permutation,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {metrics_path}")

    val = holdout.get("val", {})
    summary = {
        "subject": args.subject,
        "eeg_task": args.task,
        "top_k": args.top_k,
        "band_mode": args.bands_mode,
        "model": args.model,
        "n_features": result["n_features"],
        "n_samples": int(len(y)),
        "n_runs": len(dataset.run_ids),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "chance_accuracy": float(1 / num_classes),
        "val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "val_weighted_f1": val.get("weighted_f1"),
        "test_accuracy": holdout["test"]["accuracy"],
        "test_macro_f1": holdout["test"]["macro_f1"],
        "test_weighted_f1": holdout["test"]["weighted_f1"],
        "test_accuracy_ci_low": holdout["test"]["accuracy_ci_low"],
        "test_accuracy_ci_high": holdout["test"]["accuracy_ci_high"],
        "cv_accuracy_mean": float(accs.mean()),
        "cv_accuracy_std": float(accs.std()),
        "cv_accuracy_ci_low": cv_stats["accuracy_ci_low"],
        "cv_accuracy_ci_high": cv_stats["accuracy_ci_high"],
        "cv_macro_f1_mean": float(f1s.mean()),
        "cv_macro_f1_std": float(f1s.std()),
        "permutation_n": permutation["n"] if permutation else None,
        "permutation_null_mean": permutation["null_mean"] if permutation else None,
        "permutation_null_p95": permutation["null_p95"] if permutation else None,
        "permutation_z": permutation["z_score"] if permutation else None,
        "permutation_p": permutation["p_value"] if permutation else None,
    }
    summary_path = out_dir / f"summary_{args.subject}.csv"
    write_rows(summary_path, [summary], SUMMARY_FIELDS)
    print(f"[Saved] {summary_path}")
    return [summary]


def main():
    args = parse_args()
    subjects = list(args.subject)
    base_folds = args.folds
    run_name = args.run_name or f"baseline_{datetime.now():%Y%m%d_%H%M%S}"
    all_rows: list[dict] = []

    for subject in subjects:
        args.subject = subject
        args.folds = base_folds  # 上一个被试可能因样本太少降过折数，重置
        out_dir = args.out_dir / run_name
        if len(subjects) > 1:
            out_dir = out_dir / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== 被试 {subject} =====")
        all_rows.extend(run_subject(args, out_dir))

    run_root = args.out_dir / run_name
    if len(subjects) > 1:
        if args.sweep_bands:
            paths = sorted(run_root.glob("*/sweep_*.csv"))
            if paths:
                sweep_path = run_root / "sweep_all.csv"
                combine_csvs(paths, sweep_path)
                print(f"[Saved] {sweep_path}")
        elif all_rows:
            summary_path = run_root / "summary_all.csv"
            write_rows(summary_path, all_rows, SUMMARY_FIELDS)
            print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
