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
    uv run plan_run_identity/train.py --use-64-channels
"""

from __future__ import annotations

import argparse
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
        shuffled = rng.permutation(y)
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
    """描述 run 与原始句类的混淆程度。

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
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"), help="读取哪一组 CSV / run pkl")
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--sweep-bands", action="store_true", help="逐个频段扫描（all / 单频段 / 留一频段），Welch 只算一次")
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
    """一次完整评估：holdout + 交叉验证 + 置换零分布。"""
    model, holdout = holdout_report(
        feats, y, train_idx, {"val": val_idx, "test": test_idx}, run_names, model_name
    )
    accs, f1s = cross_validate(feats, y, model_name, folds, seed)
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


def main():
    args = parse_args()

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

    run_name = args.run_name or f"run_identity_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        classification_report(
            y[test_idx],
            model.predict(feats[test_idx]),
            labels=list(range(num_runs)),
            target_names=run_names,
            digits=3,
            zero_division=0,
        )
    )

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


if __name__ == "__main__":
    main()
