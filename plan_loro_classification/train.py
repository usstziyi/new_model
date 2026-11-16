"""leave-one-run-out 句类分类：直接检验随机划分的准确率有多少来自 run 签名。

背景：``plan_baseline_lda`` 的 8:1:1 随机划分会让同一 run 的试次同时出现在
train / test；``plan_run_identity`` 已证明 run 身份可以 98% 被解码，且 run 与
句类存在明显混杂。本实验做最直接的对照：

- **leave-one-run-out（LORO）**：每次留出整个 run 作为测试集，用其余 run
  训练，句类准确率无法再借助“见过的 run 指纹”；
- **随机划分参考**：同一批特征上跑 5 折分层 CV，复现 ``plan_baseline_lda`` 的
  within-run 数字；
- **多数类基线**：每折用训练集里最多的句类去猜测试 run，给出不依赖 EEG 的下限；
- **置换零分布**（可选）：全局打乱句类标签后重跑 LORO，给出经验 p 值；
- **频段扫描**（可选）：``all`` / 单频段 / 留一频段共 11 种特征子集。

数据直接由 ``datasets/preprocessed/<subject>_<task>.csv`` 选 top-k 句类并从原始
run pkl 加载（见 ``dataset.py``），不经过 ``top10_imagine`` 中间产物。

用法::

    uv run plan_loro_classification/train.py
    uv run plan_loro_classification/train.py --bands-mode abs
    uv run plan_loro_classification/train.py --model logreg
    uv run plan_loro_classification/train.py --use-64-channels
    uv run plan_loro_classification/train.py --sweep-bands --run-name sweep_rel_lda
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from dataset import CSV_ROOT, PKL_ROOT, LoroClassEEGDataset
from model import BANDS, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "loro_classification"


def loro_evaluate(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    run_names_by_id: dict[int, str],
    run_day: dict[int, int],
    model_name: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """逐 run 留出，返回每折指标和池化后的总体指标。"""
    splitter = LeaveOneGroupOut()
    fold_rows = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    for train_idx, test_idx in splitter.split(feats, y, groups):
        held_out = int(groups[test_idx][0])
        model = make_model(model_name)
        model.fit(feats[train_idx], y[train_idx])
        pred = model.predict(feats[test_idx])

        present = np.unique(y[test_idx])
        counts = np.bincount(y[test_idx], minlength=len(class_names))
        train_counts = np.bincount(y[train_idx], minlength=len(class_names))
        majority = int(train_counts.argmax())

        fold_rows.append(
            {
                "run": run_names_by_id[held_out],
                "run_id": held_out,
                "day": run_day.get(held_out),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "macro_f1": float(
                    f1_score(y[test_idx], pred, labels=present, average="macro", zero_division=0)
                ),
                "majority_class": class_names[majority],
                "majority_baseline_accuracy": float(np.mean(y[test_idx] == majority)),
                "class_counts": counts.tolist(),
            }
        )
        y_true_parts.append(y[test_idx])
        y_pred_parts.append(pred)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    accs = np.asarray([row["accuracy"] for row in fold_rows])
    f1s = np.asarray([row["macro_f1"] for row in fold_rows])
    majority_correct = sum(
        row["majority_baseline_accuracy"] * row["n_test"] for row in fold_rows
    )

    result = {
        "n_folds": len(fold_rows),
        "pooled": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "n_samples": int(len(y_true)),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=list(range(len(class_names)))
            ).tolist(),
            "per_class_recall": [
                float(np.mean(y_pred[y_true == c] == c)) if np.any(y_true == c) else None
                for c in range(len(class_names))
            ],
        },
        "fold_mean_accuracy": float(accs.mean()),
        "fold_std_accuracy": float(accs.std()),
        "fold_mean_macro_f1": float(f1s.mean()),
        "fold_std_macro_f1": float(f1s.std()),
        "majority_baseline_accuracy_pooled": float(majority_correct / len(y_true)),
        "folds": fold_rows,
    }
    return result, y_true, y_pred


def random_cv_reference(
    feats: np.ndarray,
    y: np.ndarray,
    model_name: str,
    folds: int,
    seed: int,
) -> dict:
    """同一批特征上的随机分层 CV，作为 within-run 参考。"""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for train_idx, test_idx in splitter.split(feats, y):
        model = make_model(model_name)
        model.fit(feats[train_idx], y[train_idx])
        pred = model.predict(feats[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    return {
        "folds": folds,
        "accuracy": accs,
        "macro_f1": f1s,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
    }


def permutation_null(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    run_names_by_id: dict[int, str],
    run_day: dict[int, int],
    model_name: str,
    seed: int,
    n_perm: int,
) -> np.ndarray:
    """全局打乱句类标签后的 LORO 池化准确率零分布。"""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(y)  # 返回打乱后的新数据
        result, _, _ = loro_evaluate(
            feats, shuffled, groups, class_names, run_names_by_id, run_day, model_name
        )
        null[i] = result["pooled"]["accuracy"]
    return null


def evaluate(
    feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
    run_names_by_id: dict[int, str],
    run_day: dict[int, int],
    model_name: str,
    folds: int,
    seed: int,
    n_perm: int,
) -> tuple[dict, np.ndarray, np.ndarray, dict, dict | None]:
    """LORO + 随机 CV 参考 (+ 置换零分布)，返回所有指标与池化预测。"""
    # 1. leave-one-run-out
    loro, y_true, y_pred = loro_evaluate(
        feats, y, groups, class_names, run_names_by_id, run_day, model_name
    )
    # 2. 同一批特征上的随机划分参考
    reference = random_cv_reference(feats, y, model_name, folds, seed)

    # 3. 置换零分布：打乱标签后重跑 LORO，p 用 +1 平滑避免为 0
    permutation = None
    if n_perm > 0:
        null = permutation_null(
            feats, y, groups, class_names, run_names_by_id, run_day,
            model_name, seed, n_perm,
        )
        observed = loro["pooled"]["accuracy"]
        permutation = {
            "n": n_perm,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "observed": float(observed),
            "z_score": float((observed - null.mean()) / (null.std() + 1e-12)),
            "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "null_samples": null.tolist(),
        }
    return loro, y_true, y_pred, reference, permutation


def parse_args():
    parser = argparse.ArgumentParser(description="leave-one-run-out 句类分类（log 频带功率 + 线性分类器）。")
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"), help="读取哪一组 CSV / run pkl")
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--sweep-bands", action="store_true", help="逐个频段扫描（all / 单频段 / 留一频段），Welch 只算一次")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器")
    parser.add_argument("--folds", type=int, default=5, help="随机划分参考的 CV 折数")
    parser.add_argument("--permutations", type=int, default=0, help="LORO 置换检验次数，0 表示跳过")
    parser.add_argument("--seed", type=int, default=42, help="随机划分参考与置换的种子")
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


def sweep_configs(channels_per_band: int) -> list[tuple[str, np.ndarray]]:
    """all + 5 个单频段 + 5 个留一频段，共 11 种列选择。"""
    bands = list(BANDS)
    configs = [("all", np.arange(len(bands) * channels_per_band))]
    configs += [(f"only:{name}", band_columns([name], None, channels_per_band)[0]) for name in bands]
    configs += [(f"drop:{name}", band_columns(None, [name], channels_per_band)[0]) for name in bands]
    return configs


def print_fold_table(rows: list[dict]) -> None:
    header = f"{'run':<9}{'day':>4}{'n_test':>8}{'acc':>8}{'macroF1':>10}{'maj_base':>10}"
    print("\n=== 逐 run 留出结果 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        day = "-" if row["day"] is None else row["day"]
        print(
            f"{row['run']:<9}{day:>4}{row['n_test']:>8}{row['accuracy']:>8.4f}"
            f"{row['macro_f1']:>10.4f}{row['majority_baseline_accuracy']:>10.4f}"
        )


def print_sweep_table(rows: list[dict]) -> None:
    """打印频段扫描的汇总表。"""
    header = (
        f"{'config':<12}{'n_feat':>7}{'loro_acc':>10}{'loro_f1':>9}"
        f"{'fold_acc':>10}{'ref_cv':>9}{'p':>8}"
    )
    print("\n=== 频段扫描汇总 ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        p_value = f"{row['p_value']:.4f}" if row["p_value"] is not None else "-"
        print(
            f"{row['config']:<12}{row['n_features']:>7}{row['loro_accuracy']:>10.4f}"
            f"{row['loro_macro_f1']:>9.4f}{row['fold_accuracy_mean']:>10.4f}"
            f"{row['reference_cv_accuracy']:>9.4f}{p_value:>8}"
        )


def base_payload(
    args,
    class_names: list[str],
    run_names: list[str],
    n_samples: int,
    n_features: int,
) -> dict:
    return {
        "subject": args.subject,
        "task": "leave_one_run_out_classification",
        "eeg_task": args.task,
        "top_k": args.top_k,
        "label_source": "meta.type_number",
        "classes": class_names,
        "runs": run_names,
        "n_samples": int(n_samples),
        "n_runs": len(run_names),
        "n_features": int(n_features),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "band_mode": args.bands_mode,
        "model": args.model,
        "sfreq": float(args.target_sfreq),
        "use_64_channels": args.use_64_channels,
    }


def run_sweep(
    feats, y, groups, class_names, run_names_by_id, run_day, run_names, args, out_dir: Path
) -> None:
    """逐个频段评估，并把汇总写入 ``sweep.json``。"""
    rows = []
    for label, columns in sweep_configs(feats.shape[1] // len(BANDS)):
        print(f"\n=== [{label}] {len(columns)} 维 ===", flush=True)
        loro, _, _, reference, permutation = evaluate(
            feats[:, columns],
            y,
            groups,
            class_names,
            run_names_by_id,
            run_day,
            args.model,
            args.folds,
            args.seed,
            args.permutations,
        )
        pooled = loro["pooled"]
        rows.append(
            {
                "config": label,
                "n_features": int(columns.size),
                "loro_accuracy": pooled["accuracy"],
                "loro_macro_f1": pooled["macro_f1"],
                "fold_accuracy_mean": loro["fold_mean_accuracy"],
                "fold_accuracy_std": loro["fold_std_accuracy"],
                "reference_cv_accuracy": reference["accuracy_mean"],
                "p_value": permutation["p_value"] if permutation else None,
                "detail": {
                    "loro": loro,
                    "reference_random_cv": reference,
                    "permutation": permutation,
                },
            }
        )
        print(
            f"  loro={pooled['accuracy']:.4f} macro_f1={pooled['macro_f1']:.4f} "
            f"fold={loro['fold_mean_accuracy']:.4f}±{loro['fold_std_accuracy']:.4f} "
            f"ref_cv={reference['accuracy_mean']:.4f}",
            flush=True,
        )

    print_sweep_table(rows)
    payload = base_payload(args, class_names, run_names, len(y), feats.shape[1])
    payload["sweep"] = rows
    sweep_path = out_dir / "sweep.json"
    with sweep_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {sweep_path}")


def main():
    args = parse_args()

    dataset = LoroClassEEGDataset(
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
    groups = dataset.groups
    class_names = list(dataset.class_names)
    run_names = list(dataset.run_names)
    run_names_by_id = dict(zip(dataset.run_ids, run_names))
    num_classes = len(class_names)
    num_runs = len(run_names)
    sfreq = float(args.target_sfreq)

    run_day: dict[int, int] = {}
    for group, entry in zip(groups, dataset.meta):
        run_day.setdefault(int(group), int(entry["day"]))

    min_class = int(np.bincount(y).min())
    if args.folds > min_class:
        print(f"[Warn] 最小的句类只有 {min_class} 条，fold 从 {args.folds} 降到 {min_class}")
        args.folds = min_class
    run_counts = np.array([np.sum(groups == run) for run in dataset.run_ids])
    print(
        f"[Runs] {num_runs} 个 run | 每 run 试次 {run_counts.min()}~{run_counts.max()} | "
        f"最小句类 {min_class} 条"
    )

    # feats: (N, 610)
    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    run_name = args.run_name or f"loro_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_bands:
        run_sweep(
            feats, y, groups, class_names, run_names_by_id, run_day, run_names, args, out_dir
        )
        return

    loro, y_true, y_pred, reference, permutation = evaluate(
        feats, y, groups, class_names, run_names_by_id, run_day,
        args.model, args.folds, args.seed, args.permutations,
    )

    # ---------- 1. leave-one-run-out ----------
    print(f"=== Leave-one-run-out（{num_runs} 折，{args.model}）===")
    print_fold_table(loro["folds"])

    pooled = loro["pooled"]
    print("\n=== LORO 汇总 ===")
    print(
        f"  池化 acc={pooled['accuracy']:.4f} macro_f1={pooled['macro_f1']:.4f} "
        f"weighted_f1={pooled['weighted_f1']:.4f} (n={pooled['n_samples']})"
    )
    print(
        f"  逐折 acc={loro['fold_mean_accuracy']:.4f} ± {loro['fold_std_accuracy']:.4f}   "
        f"macro_f1={loro['fold_mean_macro_f1']:.4f} ± {loro['fold_std_macro_f1']:.4f}"
    )
    print(
        f"  多数类基线 pooled acc={loro['majority_baseline_accuracy_pooled']:.4f}   "
        f"chance={1 / num_classes:.4f}"
    )
    print(f"\nLORO 池化分类报告（{args.model}）:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    # ---------- 2. 随机划分参考 ----------
    print(f"=== 随机 {args.folds} 折参考（within-run，同 baseline）===")
    print(
        f"  {args.model:<7} acc={reference['accuracy_mean']:.4f} ± {reference['accuracy_std']:.4f}   "
        f"macro_f1={reference['macro_f1_mean']:.4f} ± {reference['macro_f1_std']:.4f}"
    )
    print(f"  {'chance':<7} acc={1 / num_classes:.4f}")

    # ---------- 3. 置换零分布 ----------
    if permutation is not None:
        print(
            f"  null acc={permutation['null_mean']:.4f} ± {permutation['null_std']:.4f} "
            f"p95={permutation['null_p95']:.4f}   observed={permutation['observed']:.4f}   "
            f"z={permutation['z_score']:.1f}   p={permutation['p_value']:.4f}"
        )

    # ---------- 输出 ----------
    payload = base_payload(args, class_names, run_names, len(y), feats.shape[1])
    payload.update(
        {
            "loro": loro,
            "reference_random_cv": reference,
            "permutation": permutation,
        }
    )
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] {metrics_path}")


if __name__ == "__main__":
    main()
