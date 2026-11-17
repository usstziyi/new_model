"""run 身份探针：把标签换成就诊试次所在的 run，看 EEG 能否解码 run。

背景：``plan_baseline_lda`` 的句类分类用 8:1:1 随机划分，同一 run 的试次会同时
出现在 train 和 test 中。如果 run 本身在频谱特征上有强烈指纹（阻抗、疲劳、
残余伪迹等），分类器可能借 run 走捷径。本实验把标签直接改成 ``meta["run"]``，
用与 baseline 完全相同的特征和分类器跑一遍。数据直接由
``datasets/preprocessed/<subject>_<task>.csv`` 选 top-k 句类并从原始 run pkl
加载，不经过 ``top10_imagine`` 中间产物：

1. **同划分 holdout**：8:1:1 分层随机划分，报 val / test 的准确率与 macro-F1；
2. **5 折交叉验证**：全量 2459 条上的稳定估计；
3. **置换零分布**：全局打乱 run 标签后重复 5 折，给出随机水平和经验 p 值；
4. **run × 句类混淆诊断**：统计每个 run 里句类的最多数占比，以及“只用 run
   身份预测原句类”的多数类基线准确率，用来判断 run 是否同时携带句类信息。

注意：这里是**随机划分**，train/test 共享 run，因此高分只说明 run 可区分，
不等于直接证明句类分类器用了 run。若要直接检验捷径，还需要 leave-one-run-out
的句类分类对照。

用法::

    uv run plan_run_identity/train.py
    uv run plan_run_identity/train.py --bands-mode abs
    uv run plan_run_identity/train.py --permutations 100 --folds 5
    uv run plan_run_identity/train.py --sweep-bands --run-name sweep_rel_lda
    uv run plan_run_identity/train.py --sweep-channels --run-name sweep_channels_lda
    uv run plan_run_identity/train.py --use-64-channels
    uv run plan_run_identity/train.py --subject sub-01 sub-02
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

from dataset import CSV_ROOT, PKL_ROOT, RunLabelEEGDataset, split_indices
from model import BANDS, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "run_identity"
MONTAGE_CSV = PROJECT_ROOT / "scripts/montage.csv"
BAD_CHANNELS_CSV = PROJECT_ROOT / "outputs" / "bad_channels" / "bad_channels.csv"


def cross_validate(X, y, model_name: str, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """返回每折的 (accuracy, macro_f1)。"""
    accs, f1s = [], []
    # 分层 K 折：每折的 run 比例与全集一致
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        model = make_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    return np.asarray(accs), np.asarray(f1s)


def permutation_null(X, y, model_name: str, folds: int, seed: int, n_perm: int) -> np.ndarray:
    """全局打乱 run 标签后的 CV 准确率零分布。"""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(y)  # 返回打乱后的新数据
        accs, _ = cross_validate(X, shuffled, model_name, folds, seed)
        null[i] = accs.mean()
    return null


def holdout_report(X, y, train_idx, eval_sets, run_names, model_name):
    """在固定划分上训练一次，返回各评估集的指标。"""
    model = make_model(model_name)
    model.fit(X[train_idx], y[train_idx])
    out = {}
    for name, idx in eval_sets.items():
        pred = model.predict(X[idx])
        out[name] = {
            "accuracy": float(accuracy_score(y[idx], pred)),
            "macro_f1": float(f1_score(y[idx], pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[idx], pred, average="weighted", zero_division=0)),
            "n_samples": int(len(idx)),
            "confusion_matrix": confusion_matrix(
                y[idx], pred, labels=list(range(len(run_names)))
            ).tolist(),
            "per_class_recall": [
                float(np.mean(pred[y[idx] == c] == c)) if np.any(y[idx] == c) else None
                for c in range(len(run_names))
            ],
        }
    return model, out


def run_class_confound(run_labels, orig_labels, run_names, orig_class_names, meta) -> dict:
    """描述 run 与原始句类的混淆程度。它衡量的是“数据中 run 和句类纠缠得有多严重”。

    对每个 run 统计其试次的句类分布；``majority_baseline_accuracy`` 表示只看
    run 身份、统一猜该 run 里最常见的句类时的准确率。
    """
    total_majority = 0
    rows = []
    for run_label, run_name in enumerate(run_names):
        mask = run_labels == run_label
        counts = np.bincount(orig_labels[mask], minlength=len(orig_class_names))
        majority = int(counts.argmax())
        total_majority += int(counts[majority])
        rows.append(
            {
                "run": run_name,
                "day": sorted({int(meta[i].get("day", -1)) for i in np.flatnonzero(mask)}),
                "n_trials": int(mask.sum()),
                "majority_class": orig_class_names[majority],
                "majority_share": float(counts[majority] / mask.sum()),
            }
        )
    return {
        "n_runs": len(run_names),
        "n_original_classes": len(orig_class_names),
        "majority_baseline_accuracy": float(total_majority / len(run_labels)),
        "random_baseline_accuracy": float(1.0 / len(orig_class_names)),
        "mean_majority_share": float(np.mean([row["majority_share"] for row in rows])),
        "runs": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="run 身份探针：log 频带功率 + 线性分类器。")
    parser.add_argument("--subject", nargs="+", default=["sub-01"], help="被试编号，可指定多个，如 --subject sub-01 sub-02")
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"), help="读取哪一组 CSV / run pkl")
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--sweep-bands", action="store_true", help="逐个频段扫描（all / 单频段 / 留一频段），Welch 只算一次")
    parser.add_argument("--sweep-channels", action="store_true", help="逐通道扫描：run 效应量排序 + 首尾通道的 only/drop CV")
    parser.add_argument("--channel-top-k", type=int, default=10, help="通道扫描确认阶段取效应量最高/最低的各 k 个")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器：lda 收缩 LDA、logreg 逻辑回归或 svm RBF 核 SVM")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--permutations", type=int, default=200, help="置换检验次数，0 表示跳过")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--split-seed", type=int, default=42, help="划分种子")
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


def channel_columns(n_channels: int, channel: int) -> np.ndarray:
    """返回某通道的列索引（特征按 ``[频段][通道]`` 主序拼接）。"""
    return np.arange(len(BANDS)) * n_channels + channel


def channel_eta_squared(feats: np.ndarray, y: np.ndarray, n_channels: int) -> np.ndarray:
    """每个 (通道, 频段) 的 run 组间效应量 eta^2，返回 ``(n_channels, n_bands)``。"""
    n_bands = len(BANDS)
    x = feats.reshape(len(feats), n_bands, n_channels)
    runs = np.unique(y)
    onehot = np.zeros((len(y), len(runs)))
    onehot[np.arange(len(y)), np.searchsorted(runs, y)] = 1
    counts = onehot.sum(axis=0)
    flat = x.reshape(len(feats), -1)
    means = (onehot.T @ flat / counts[:, None]).reshape(len(runs), n_bands, n_channels)
    grand = flat.mean(axis=0).reshape(n_bands, n_channels)
    ss_between = (counts[:, None, None] * (means - grand) ** 2).sum(axis=0)
    ss_total = ((x - grand) ** 2).sum(axis=0)
    return (ss_between / np.maximum(ss_total, 1e-30)).T


def load_montage(path: Path = MONTAGE_CSV) -> dict[str, dict[str, float]]:
    """读取电极坐标，键为小写通道名；文件缺失时返回空字典。"""
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["label"].lower(): {axis: float(row[axis]) for axis in ("x", "y", "z")}
        for row in rows
        if row.get("label")
    }


def load_bad_run_counts(subject: str, task: str, path: Path = BAD_CHANNELS_CSV) -> dict[str, int]:
    """统计每个通道在多少次 run 中被判为坏道；文件缺失时返回空字典。"""
    if not Path(path).exists():
        return {}
    counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("subject") != subject or row.get("task") != task:
                continue
            for name in (row.get("bad_channels") or "").split(";"):
                name = name.strip()
                if name:
                    counts[name] = counts.get(name, 0) + 1
    return counts


def evaluate(
    feats: np.ndarray,
    y: np.ndarray,
    run_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
    n_perm: int,
) -> tuple:
    
    # 1. “留出法”即在固定的 8:1:1 划分上训练一次，报告 val / test 的 holdout 指标
    model, holdout = holdout_report(
        feats, y, train_idx, {"val": val_idx, "test": test_idx}, run_names, model_name
    )
    # 2. 全量数据 5 折交叉验证，其均值作为置换检验的观测值
    accs, f1s = cross_validate(feats, y, model_name, folds, seed)
    permutation = None
    if n_perm > 0:
        # 3. 置换零分布：打乱标签后重跑同一条管线，p 用 +1 平滑避免为 0（下限 1/(n+1)）
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
        "cv": {model_name: {"accuracy": accs.tolist(), "macro_f1": f1s.tolist()}},
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
        f"{'config':<12}{'n_feat':>7}{'val':>8}{'test':>8}"
        f"{'cv_acc':>9}{'std':>7}{'macroF1':>9}{'p':>8}"
    )
    print("\n=== 频段扫描汇总 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        p_value = f"{row['p_value']:.4f}" if row["p_value"] is not None else "-"
        print(
            f"{row['config']:<12}{row['n_features']:>7}{row['val_accuracy']:>8.4f}"
            f"{row['test_accuracy']:>8.4f}{row['cv_accuracy_mean']:>9.4f}"
            f"{row['cv_accuracy_std']:>7.4f}{row['cv_macro_f1_mean']:>9.4f}{p_value:>8}"
        )


def sweep_payload(args, run_names, orig_class_names, confound, train_idx, val_idx, test_idx, rows):
    return {
        "subject": args.subject,
        "task": "run_identity",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "label_source": "meta.run",
        "runs": run_names,
        "original_classes": orig_class_names,
        "n_samples": int(len(train_idx) + len(val_idx) + len(test_idx)),
        "n_runs": len(run_names),
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
        "run_class_confound": confound,
        "sweep": rows,
    }


def run_sweep(feats, y, run_names, orig_class_names, confound, train_idx, val_idx, test_idx, args, out_dir: Path) -> None:
    """逐个频段评估，并把汇总写入 ``sweep.json``。"""
    rows = []
    for label, columns in sweep_configs(feats.shape[1] // len(BANDS)):
        print(f"\n=== [{label}] {len(columns)} 维 ===", flush=True)
        _, result = evaluate(
            feats[:, columns],
            y,
            run_names,
            train_idx,
            val_idx,
            test_idx,
            args.model,
            args.folds,
            args.seed,
            args.permutations,
        )
        holdout = result["holdout"]
        accs = np.asarray(result["cv"][args.model]["accuracy"])
        f1s = np.asarray(result["cv"][args.model]["macro_f1"])
        permutation = result["permutation"]
        rows.append(
            {
                "config": label,
                "n_features": result["n_features"],
                "val_accuracy": holdout["val"]["accuracy"],
                "test_accuracy": holdout["test"]["accuracy"],
                "cv_accuracy_mean": float(accs.mean()),
                "cv_accuracy_std": float(accs.std()),
                "cv_macro_f1_mean": float(f1s.mean()),
                "p_value": permutation["p_value"] if permutation else None,
                "detail": result,
            }
        )
        print(
            f"  val={rows[-1]['val_accuracy']:.4f} test={rows[-1]['test_accuracy']:.4f} "
            f"cv={rows[-1]['cv_accuracy_mean']:.4f}±{rows[-1]['cv_accuracy_std']:.4f}",
            flush=True,
        )

    print_sweep_table(rows)
    payload = sweep_payload(
        args, run_names, orig_class_names, confound, train_idx, val_idx, test_idx, rows
    )
    sweep_path = out_dir / "sweep.json"
    with sweep_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {sweep_path}")


def run_channel_sweep(feats, y, run_names, channel_names, confound, train_idx, val_idx, test_idx, args, out_dir: Path) -> None:
    """按通道计算 run 效应量，并对首尾通道做 only/drop 确认。"""
    n_channels = len(channel_names)
    band_names = list(BANDS)
    eta = channel_eta_squared(feats, y, n_channels)
    score = eta.mean(axis=1)
    order = np.argsort(-score)
    montage = load_montage()
    bad_counts = load_bad_run_counts(args.subject, args.task)

    channels = []
    for rank, channel in enumerate(order, start=1):
        name = channel_names[channel]
        entry = {
            "rank": rank,
            "index": int(channel),
            "name": name,
            "eta2_mean": float(score[channel]),
            "eta2_max": float(eta[channel].max()),
            "eta2": {band: float(eta[channel, i]) for i, band in enumerate(band_names)},
            "bad_run_count": int(bad_counts.get(name, 0)),
        }
        entry.update(montage.get(name.lower(), {}))
        channels.append(entry)

    print(f"\n=== 通道效应量 Top {min(20, n_channels)}（按 run 间 eta^2 均值）===")
    header = f"{'rank':>4}  {'channel':<8}{'eta2_mean':>10}{'eta2_max':>10}{'bad_runs':>9}{'best_band':>11}"
    print(header)
    print("-" * len(header))
    for entry in channels[:20]:
        best_band = max(entry["eta2"], key=entry["eta2"].get)
        print(
            f"{entry['rank']:>4}  {entry['name']:<8}{entry['eta2_mean']:>10.4f}"
            f"{entry['eta2_max']:>10.4f}{entry['bad_run_count']:>9}{best_band:>11}"
        )

    top_k = min(max(args.channel_top_k, 0), n_channels)
    # 最高 / 最低各 top_k 个，分组输出并按 eta^2 降序
    groups = [("top", order[:top_k]), ("bottom", order[::-1][:top_k])]

    _, baseline = evaluate(
        feats, y, run_names, train_idx, val_idx, test_idx,
        args.model, args.folds, args.seed, 0,
    )
    base_acc = float(np.mean(baseline["cv"][args.model]["accuracy"]))
    print(f"\n=== 通道确认：only-channel / drop-channel（all={base_acc:.4f}）===")
    header = f"{'channel':<8}{'eta2':>8}{'only_cv':>9}{'drop_cv':>9}{'only_test':>10}{'drop_test':>10}"
    print(header)
    print("-" * len(header))
    confirm = []
    seen = set()
    for group, indices in groups:
        if top_k == 0:
            break
        print(f"[{group}]")
        for channel in indices:
            channel = int(channel)
            if channel in seen:  # top 与 bottom 重叠时只评估一次
                continue
            seen.add(channel)
            columns = channel_columns(n_channels, channel)
            keep = np.ones(feats.shape[1], dtype=bool)
            keep[columns] = False
            _, only = evaluate(
                feats[:, columns], y, run_names, train_idx, val_idx, test_idx,
                args.model, args.folds, args.seed, 0,
            )
            _, drop = evaluate(
                feats[:, keep], y, run_names, train_idx, val_idx, test_idx,
                args.model, args.folds, args.seed, 0,
            )
            only_acc = float(np.mean(only["cv"][args.model]["accuracy"]))
            drop_acc = float(np.mean(drop["cv"][args.model]["accuracy"]))
            row = {
                "group": group,
                "channel": channel_names[channel],
                "index": channel,
                "eta2_mean": float(score[channel]),
                "only_cv_accuracy": only_acc,
                "only_test_accuracy": float(only["holdout"]["test"]["accuracy"]),
                "drop_cv_accuracy": drop_acc,
                "drop_test_accuracy": float(drop["holdout"]["test"]["accuracy"]),
                "drop_delta_cv": drop_acc - base_acc,
            }
            confirm.append(row)
            print(
                f"{row['channel']:<8}{row['eta2_mean']:>8.4f}{only_acc:>9.4f}{drop_acc:>9.4f}"
                f"{row['only_test_accuracy']:>10.4f}{row['drop_test_accuracy']:>10.4f}"
            )

    payload = {
        "subject": args.subject,
        "task": "run_identity_channel_sweep",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "label_source": "meta.run",
        "n_samples": int(len(y)),
        "n_runs": len(run_names),
        "n_channels": n_channels,
        "n_bands": len(band_names),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": float(args.target_sfreq),
        "use_64_channels": args.use_64_channels,
        "feature_layout": "按 [频段][通道] 主序拼接",
        "selection_metric": "run 组间 eta^2（逐通道逐频段）",
        "split": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.split_seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "run_class_confound": confound,
        "top_k_channels": top_k,
        "baseline_all": {
            "cv_accuracy_mean": base_acc,
            "test_accuracy": float(baseline["holdout"]["test"]["accuracy"]),
        },
        "channels": channels,
        "confirm": confirm,
    }
    path = out_dir / "channel_sweep.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {path}")


REPORT_FIELDS = ["subject", "label", "precision", "recall", "f1", "support"]


def save_classification_csv(path: Path, subject: str, report: dict) -> None:
    """把 ``classification_report(output_dict=True)`` 的逐 run 指标写成 CSV。"""
    total_support = sum(
        metrics["support"]
        for label, metrics in report.items()
        if isinstance(metrics, dict) and label not in ("macro avg", "weighted avg", "samples avg")
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
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
            writer.writerow(row)


def save_summary_csv(path: Path, row: dict) -> None:
    """把单个被试的总体指标写成一行 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run_subject(args, out_dir: Path) -> None:
    dataset = RunLabelEEGDataset(
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
    run_names = list(dataset.run_names)
    num_runs = len(run_names)
    orig_class_names = list(dataset.orig_class_names)
    sfreq = float(args.target_sfreq)

    min_class = int(np.bincount(y).min())
    if args.folds > min_class:
        print(f"[Warn] 最小的 run 只有 {min_class} 条，fold 从 {args.folds} 降到 {min_class}")
        args.folds = min_class
    print(f"[Runs] {num_runs} 个 run | 每 run 试次 {min_class}~{int(np.bincount(y).max())}")

    # 混淆诊断
    confound = run_class_confound(
        y, dataset.orig_labels, run_names, orig_class_names, dataset.meta
    )
    print(
        f"[Confound] 只用 run 身份猜原句类：多数类基线 acc="
        f"{confound['majority_baseline_accuracy']:.4f} "
        f"(随机 {confound['random_baseline_accuracy']:.4f}) | "
        f"平均多数类占比={confound['mean_majority_share']:.4f}"
    )

    # feats: (N, 610)
    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    # ---------- 1. 8:1:1 分层随机划分 ----------
    train_idx, val_idx, test_idx = split_indices(
        y, args.val_ratio, args.test_ratio, args.split_seed
    )
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    if args.sweep_bands and args.sweep_channels:
        raise SystemExit("--sweep-bands 与 --sweep-channels 只能选一个")

    if args.sweep_channels:
        run_channel_sweep(
            feats, y, run_names, list(dataset.channel_names), confound,
            train_idx, val_idx, test_idx, args, out_dir,
        )
        return

    if args.sweep_bands:
        run_sweep(
            feats, y, run_names, orig_class_names, confound,
            train_idx, val_idx, test_idx, args, out_dir,
        )
        return

    model, result = evaluate(
        feats, y, run_names, train_idx, val_idx, test_idx,
        args.model, args.folds, args.seed, args.permutations,
    )
    holdout = result["holdout"]
    for name in ("val", "test"):
        m = holdout[name]
        print(
            f"  [{args.model}] {name:<4} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f} (n={m['n_samples']})"
        )
    print(f"\n测试集分类报告（{args.model}，标签=run）:")
    pred_test = model.predict(feats[test_idx])
    report_dict = classification_report(
        y[test_idx],
        pred_test,
        labels=list(range(num_runs)),
        target_names=run_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            y[test_idx],
            pred_test,
            labels=list(range(num_runs)),
            target_names=run_names,
            digits=3,
            zero_division=0,
        )
    )
    report_path = out_dir / f"classification_report_{args.subject}.csv"
    save_classification_csv(report_path, args.subject, report_dict)
    print(f"[Saved] {report_path}")

    # ---------- 2. 交叉验证 ----------
    print(f"=== {args.folds} 折交叉验证（全量 {len(y)} 条）===")
    accs = np.asarray(result["cv"][args.model]["accuracy"])
    f1s = np.asarray(result["cv"][args.model]["macro_f1"])
    for i, (acc, f1) in enumerate(zip(accs, f1s), start=1):
        print(f"  fold {i}  acc={acc:.4f}  macro_f1={f1:.4f}")
    print(
        f"  {args.model:<7} acc={accs.mean():.4f} ± {accs.std():.4f}   "
        f"macro_f1={f1s.mean():.4f} ± {f1s.std():.4f}"
    )
    print(f"  {'chance':<7} acc={1 / num_runs:.4f}")

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
        "task": "run_identity",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "label_source": "meta.run",
        "runs": run_names,
        "original_classes": orig_class_names,
        "n_samples": int(len(y)),
        "n_runs": num_runs,
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
        "run_class_confound": confound,
        "holdout": holdout,
        "cv": result["cv"],
        "permutation": permutation,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {metrics_path}")

    summary = {
        "subject": args.subject,
        "eeg_task": args.task,
        "top_k": args.top_k,
        "band_mode": args.bands_mode,
        "model": args.model,
        "n_features": result["n_features"],
        "n_samples": int(len(y)),
        "n_runs": num_runs,
        "chance_accuracy": float(1 / num_runs),
        "val_accuracy": holdout["val"]["accuracy"],
        "val_macro_f1": holdout["val"]["macro_f1"],
        "val_weighted_f1": holdout["val"]["weighted_f1"],
        "test_accuracy": holdout["test"]["accuracy"],
        "test_macro_f1": holdout["test"]["macro_f1"],
        "test_weighted_f1": holdout["test"]["weighted_f1"],
        "cv_accuracy_mean": float(accs.mean()),
        "cv_accuracy_std": float(accs.std()),
        "cv_macro_f1_mean": float(f1s.mean()),
        "cv_macro_f1_std": float(f1s.std()),
        "permutation_n": permutation["n"] if permutation else None,
        "permutation_null_mean": permutation["null_mean"] if permutation else None,
        "permutation_null_p95": permutation["null_p95"] if permutation else None,
        "permutation_z": permutation["z_score"] if permutation else None,
        "permutation_p": permutation["p_value"] if permutation else None,
        "run_majority_baseline_accuracy": confound["majority_baseline_accuracy"],
        "run_mean_majority_share": confound["mean_majority_share"],
    }
    summary_path = out_dir / f"summary_{args.subject}.csv"
    save_summary_csv(summary_path, summary)
    print(f"[Saved] {summary_path}")


def main():
    args = parse_args()
    subjects = list(args.subject)
    base_folds = args.folds
    run_name = args.run_name or f"run_identity_{datetime.now():%Y%m%d_%H%M%S}"

    for subject in subjects:
        args.subject = subject
        args.folds = base_folds  # 上一个被试可能因样本太少降过折数，重置
        out_dir = args.out_dir / run_name
        if len(subjects) > 1:
            out_dir = out_dir / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== 被试 {subject} =====")
        run_subject(args, out_dir)


if __name__ == "__main__":
    main()
