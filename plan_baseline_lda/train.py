"""训练并评估线性基线：log 频带功率 + 收缩 LDA / 逻辑回归 / RBF SVM。

深度模型的验证/测试指标必须和这里的数字比较，过不了这条线就不该继续加大模型。

评估内容：

1. **同划分对照**：用 ``dataset.split_indices`` 复现 ``plan_top10_random/train.py``
   的 8:1:1 划分，在 val / test 上给出准确率与 macro-F1，可直接和 PMCNN 的
   ``history.csv`` 对比；
2. **5 折交叉验证**：全量 2459 条上的稳定估计（均值 ± 标准差）；
3. **置换零分布**：全局打乱标签后重复 5 折，给出随机水平与经验 p 值；
4. **频段扫描**（``--sweep-bands``）：all / 单频段 / 留一频段共 11 种配置，Welch 只算一次。

特征与分类器见 ``model.py``：逐通道 5 个频段的 log 功率（默认相对功率），


相对功率对逐试次 z-score 不敏感，因此这里关闭 ``normalize``，直接在 µV 上做 Welch。
分类器由 ``--model`` 选择，默认收缩 LDA。

用法::


    uv run plan_baseline_lda/train.py --sweep-bands --run-name sweep_rel_lda


    uv run plan_baseline_lda/train.py
    uv run plan_baseline_lda/train.py --model svm      # RBF 核 SVM
    uv run plan_baseline_lda/train.py --model logreg
    uv run plan_baseline_lda/train.py --use-64-channels
    uv run plan_baseline_lda/train.py --permutations 100 --folds 5
    uv run plan_baseline_lda/train.py --bands-mode abs      # 改用绝对 log 功率
    uv run plan_baseline_lda/train.py --sweep-bands         # 单频段 + 留一频段扫描
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from dataset import TOP_ROOT, TopClassEEGDataset, split_indices
from model import BANDS, extract_features, make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baseline_lda"


def cross_validate(X, y, model_name: str, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """返回每折的 (accuracy, macro_f1)。"""
    accs, f1s = [], []
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
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    return np.asarray(accs), np.asarray(f1s)


def permutation_null(X, y, model_name: str, folds: int, seed: int, n_perm: int) -> np.ndarray:
    """全局打乱标签后的 CV 准确率零分布。"""
    rng = np.random.default_rng(seed)
    # null 来自统计学的 null hypothesis（原假设或零假设）
    null = np.empty(n_perm)
    for i in range(n_perm):
        # 置换操作：全局打乱标签
        shuffled = rng.permutation(y)
        accs, _ = cross_validate(X, shuffled, model_name, folds, seed)
        null[i] = accs.mean()

    return null


def holdout_report(X, y, train_idx, eval_sets, class_names, model_name):
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
    parser.add_argument("--subject", default="sub-01", help="被试编号")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="降采样目标采样率")
    parser.add_argument("--use-64-channels", action="store_true", help="只用 64 通道")
    parser.add_argument("--bands-mode", default="rel", choices=("rel", "abs"), help="相对/绝对频带功率")
    parser.add_argument("--sweep-bands", action="store_true", help="逐个频段扫描（all / 单频段 / 留一频段），Welch 只算一次")
    parser.add_argument("--model", default="lda", choices=("lda", "logreg", "svm"), help="分类器：lda 收缩 LDA（主基线）、logreg 逻辑回归或 svm RBF 核 SVM")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--permutations", type=int, default=200, help="置换检验次数，0 表示跳过")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例（与 train.py 一致）")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例（与 train.py 一致）")
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
    model, holdout = holdout_report(
        feats, y, train_idx, {"val": val_idx, "test": test_idx}, class_names, model_name
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


def run_sweep(feats, y, class_names, train_idx, val_idx, test_idx, args, out_dir: Path) -> None:
    """逐个频段评估，并把汇总写入 ``sweep.json``。"""
    rows = []
    for label, columns in sweep_configs(feats.shape[1] // len(BANDS)):
        print(f"\n=== [{label}] {len(columns)} 维 ===", flush=True)
        _, result = evaluate(
            feats[:, columns],
            y,
            class_names,
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
    payload = {
        "subject": args.subject,
        "classes": class_names,
        "n_samples": int(len(y)),
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


def main():
    args = parse_args()

    dataset = TopClassEEGDataset(
        subject=args.subject,
        root=args.data_root,
        target_sfreq=args.target_sfreq,
        normalize=False,  # 相对功率尺度不变，无需逐试次 z-score
        use_64_channels=args.use_64_channels,
    )
    y = dataset.labels
    class_names = list(dataset.class_names)
    num_classes = len(class_names)
    sfreq = float(args.target_sfreq)

    # feats: (N, 610)
    feats = extract_features(dataset.data.numpy(), sfreq, args.bands_mode)
    channels_per_band = feats.shape[1] // len(BANDS)
    print(
        f"[Features] {args.bands_mode} log 频带功率 | {feats.shape[1]} 维 "
        f"({channels_per_band} 通道 × {len(BANDS)} 频段) | sfreq={sfreq:g} Hz",
        flush=True,
    )

    # ---------- 1. 与 train.py 相同的 8:1:1 划分 ----------
    train_idx, val_idx, test_idx = split_indices(
        y, args.val_ratio, args.test_ratio, args.split_seed
    )
    print(f"[Split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    run_name = args.run_name or f"baseline_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_bands:
        run_sweep(feats, y, class_names, train_idx, val_idx, test_idx, args, out_dir)
        return

    model, result = evaluate(
        feats, y, class_names, train_idx, val_idx, test_idx,
        args.model, args.folds, args.seed, args.permutations,
    )
    holdout = result["holdout"]
    for name in ("val", "test"):
        m = holdout[name]
        print(
            f"  [{args.model}] {name:<4} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f} (n={m['n_samples']})"
        )
    print(f"\n测试集分类报告（{args.model}）:")
    print(
        classification_report(
            y[test_idx],
            model.predict(feats[test_idx]),
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    # ---------- 2. 5 折交叉验证 ----------
    print(f"=== {args.folds} 折交叉验证（全量 {len(y)} 条）===")
    accs = np.asarray(result["cv"][args.model]["accuracy"])
    f1s = np.asarray(result["cv"][args.model]["macro_f1"])
    for i, (acc, f1) in enumerate(zip(accs, f1s), start=1):
        print(f"  fold {i}  acc={acc:.4f}  macro_f1={f1:.4f}")
    print(
        f"  {args.model:<7} acc={accs.mean():.4f} ± {accs.std():.4f}   "
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
        "classes": class_names,
        "n_samples": int(len(y)),
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


if __name__ == "__main__":
    main()
