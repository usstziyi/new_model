"""EEGNet 表征的 run 身份线性探针（方案 A：单次训练 + 全量特征探针）。

问题：句类 LORO 掉到 chance，是"EEGNet 没有编码 run"，还是"它编码了 run 但句类
信号本身不存在"？本脚本直接在**训练好的 EEGNet 表征**上做线性 run 分类，作为
``experiments/diagnostics/run_identity``（原始 610 维频带功率，98.4%）的同构对照。

流程：

1. 用与 ``train.py --protocol holdout`` 完全相同的 8:2 分层划分（同 split_seed）
   训练一次 EEGNet，训练轮数固定（默认 30，不做基于测试集的早停）；
2. 用 ``EEGNet.embed`` 提取全部试次的倒数第二层特征（默认 816 维）；
3. 在该特征上做两组线性探针，分类器与主实验一致（标准化 + 收缩 LDA）：

   - ``probe_all``：全部 2,459 条样本上的 5 折分层交叉验证，标签为 run。
     这是与"原始频带功率 98.4%"可比的口径；
   - ``probe_test``：只用留出的 20% 样本（约 492 条）做 5 折交叉验证。
     这些特征来自未见过该样本的模型，可排除"记住训练样本"的解释。

4. 另做一组负向对照 ``class_probe_test``：同样的留出特征、同样的分类器，
   标签换成句类，用于说明"run 信息破解码"不等于"句类可迁移"。

注意：``probe_all`` 中训练集样本的特征由见过它们的模型产生，可能通过记忆而非
系统性编码携带 run 信息；因此两组探针必须并列报告，不能只报前者。

用法::

    uv run experiments/deepnet/eegnet/probe.py --subject sub-01
    uv run experiments/deepnet/eegnet/probe.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 \
        --run-name main_probe
    uv run experiments/deepnet/eegnet/probe.py --subject sub-01 --epochs 60 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dataset import CSV_ROOT, PKL_ROOT, EEGNetDataset, split_indices
from model import build_model, count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "deepnet" / "eegnet" / "run_probe"

PROBE_FIELDS = [
    "subject",
    "probe",
    "n_samples",
    "n_classes",
    "n_folds",
    "n_features",
    "accuracy",
    "accuracy_std",
    "macro_f1",
    "chance",
    "majority_baseline",
    "epochs",
    "n_params",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但当前环境没有可用 GPU")
    return device


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class XOnly(Dataset):
    """只返回样本、不带标签的包装，用于特征提取。"""

    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        x, _ = self.base[int(self.indices[index])]
        return x


# ---------------------------------------------------------------------------
# 训练与特征提取
# ---------------------------------------------------------------------------
def train_classifier(
    args,
    dataset: EEGNetDataset,
    train_idx: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """在 ``train_idx`` 上训练 EEGNet 句类分类器，返回训练好的模型。"""
    set_seed(args.seed)
    model = build_model(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        n_classes=num_classes,
        sfreq=dataset.sfreq,
        f1=args.f1,
        d=args.d,
        f2=args.f2,
        kernel_seconds=args.kernel_seconds,
        separable_seconds=args.separable_seconds,
        dropout=args.dropout,
        dropout2=args.dropout2,
    ).to(device)

    train_set = XOnlyLabels(dataset, train_idx, dataset.labels)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += len(y)
        scheduler.step()
        if not args.quiet_steps:
            print(
                f"  epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}",
                flush=True,
            )
    return model


class XOnlyLabels(Dataset):
    """按外部标签覆盖 base 标签的取样子集（训练用）。"""

    def __init__(self, base: Dataset, indices: np.ndarray, labels: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        base_index = int(self.indices[index])
        x, _ = self.base[base_index]
        return x, int(self.labels[base_index])


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset: EEGNetDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """提取 ``indices`` 对应样本的倒数第二层特征，返回 ``(N, D)`` float32。"""
    model.eval()
    loader = DataLoader(
        XOnly(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    parts: list[np.ndarray] = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        parts.append(model.embed(x).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 线性探针
# ---------------------------------------------------------------------------
def make_probe():
    """与主实验一致的线性分类器：标准化 + 收缩 LDA。"""
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def run_probe(
    feats: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """在给定特征上做分层 K 折线性探针。

    Returns:
        ``(pooled_accuracy, fold_std, macro_f1, n_folds_effective)``
    """
    labels = np.asarray(labels)
    min_count = int(np.bincount(labels).min())
    n_folds = max(2, min(folds, min_count))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    accs: list[float] = []
    for train_idx, test_idx in splitter.split(feats, labels):
        model = make_probe()
        model.fit(feats[train_idx], labels[train_idx])
        pred = model.predict(feats[test_idx])
        y_true_parts.append(labels[test_idx])
        y_pred_parts.append(pred)
        accs.append(accuracy_score(labels[test_idx], pred))
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(np.std(accs)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        n_folds,
    )


# ---------------------------------------------------------------------------
# 单被试入口
# ---------------------------------------------------------------------------
def run_subject(
    args, dataset: EEGNetDataset, out_dir: Path, device: torch.device
) -> list[dict]:
    y = dataset.labels
    num_classes = len(dataset.class_ids)
    run_ids = sorted({int(run) for run in dataset.run_groups})
    run_to_label = {run: index for index, run in enumerate(run_ids)}
    y_run = np.asarray([run_to_label[int(run)] for run in dataset.run_groups], dtype=np.int64)

    train_idx, _, test_idx = split_indices(y, 0.0, 0.2, args.split_seed)
    print(
        f"[Split] seed={args.split_seed} train={len(train_idx)} test={len(test_idx)}",
        flush=True,
    )

    print(f"[Train] EEGNet {args.epochs} epoch，仅用训练集", flush=True)
    model = train_classifier(args, dataset, train_idx, num_classes, device)
    n_params = count_parameters(model)

    print("[Features] 提取全部试次的倒数第二层特征", flush=True)
    all_idx = np.arange(len(y))
    feats = extract_features(
        model, dataset, all_idx, device, args.batch_size, args.num_workers
    )
    print(f"[Features] x {tuple(feats.shape)}", flush=True)

    rows: list[dict] = []
    details: dict[str, dict] = {}

    def collect(name: str, labels: np.ndarray, n_classes: int, note: str) -> None:
        acc, std, f1, n_folds = run_probe(feats, labels, args.folds, args.seed)
        majority = float(np.bincount(labels).max() / len(labels))
        row = {
            "subject": args.subject,
            "probe": name,
            "n_samples": int(len(labels)),
            "n_classes": int(n_classes),
            "n_folds": int(n_folds),
            "n_features": int(feats.shape[1]),
            "accuracy": acc,
            "accuracy_std": std,
            "macro_f1": f1,
            "chance": float(1 / n_classes),
            "majority_baseline": majority,
            "epochs": args.epochs,
            "n_params": n_params,
        }
        rows.append(row)
        details[name] = {**row, "note": note}
        print(
            f"[{name}] acc={acc:.4f} ± {std:.4f} | chance={1/n_classes:.4f} "
            f"| macro_f1={f1:.4f} | n={len(labels)} 类数={n_classes}",
            flush=True,
        )

    # 1) 全量样本上的 run 探针（与原始频带功率 98.4% 同口径）
    collect("probe_all_run", y_run, len(run_ids), "全部样本，标签为 run")
    # 2) 只用留出样本的 run 探针（特征来自未见过该样本的模型）
    test_feats = feats[test_idx]
    acc, std, f1, n_folds = run_probe(
        test_feats, y_run[test_idx], args.folds, args.seed
    )
    majority = float(np.bincount(y_run[test_idx]).max() / len(test_idx))
    row = {
        "subject": args.subject,
        "probe": "probe_test_run",
        "n_samples": int(len(test_idx)),
        "n_classes": int(len(run_ids)),
        "n_folds": int(n_folds),
        "n_features": int(test_feats.shape[1]),
        "accuracy": acc,
        "accuracy_std": std,
        "macro_f1": f1,
        "chance": float(1 / len(run_ids)),
        "majority_baseline": majority,
        "epochs": args.epochs,
        "n_params": n_params,
    }
    rows.append(row)
    details["probe_test_run"] = {**row, "note": "仅留出 20% 样本，标签为 run"}
    print(
        f"[probe_test_run] acc={acc:.4f} ± {std:.4f} | chance={1/len(run_ids):.4f} "
        f"| macro_f1={f1:.4f} | n={len(test_idx)}",
        flush=True,
    )

    # 3) 负向对照：同样的留出特征，标签换成句类
    acc, std, f1, n_folds = run_probe(test_feats, y[test_idx], args.folds, args.seed)
    majority = float(np.bincount(y[test_idx]).max() / len(test_idx))
    row = {
        "subject": args.subject,
        "probe": "probe_test_class",
        "n_samples": int(len(test_idx)),
        "n_classes": int(num_classes),
        "n_folds": int(n_folds),
        "n_features": int(test_feats.shape[1]),
        "accuracy": acc,
        "accuracy_std": std,
        "macro_f1": f1,
        "chance": float(1 / num_classes),
        "majority_baseline": majority,
        "epochs": args.epochs,
        "n_params": n_params,
    }
    rows.append(row)
    details["probe_test_class"] = {**row, "note": "仅留出 20% 样本，标签为句类（负向对照）"}
    print(
        f"[probe_test_class] acc={acc:.4f} ± {std:.4f} | chance={1/num_classes:.4f} "
        f"| macro_f1={f1:.4f} | n={len(test_idx)}",
        flush=True,
    )

    payload = {
        "subject": args.subject,
        "task": "eegnet_run_probe",
        "eeg_task": args.task,
        "dataset": {
            "n_samples": int(len(y)),
            "n_channels": int(dataset.n_channels),
            "n_times": int(dataset.n_times),
            "sfreq": float(dataset.sfreq),
            "n_runs": len(run_ids),
            "n_classes": num_classes,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_params": n_params,
        },
        "probes": details,
    }
    with (out_dir / "probe_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_rows(out_dir / f"probe_summary_{args.subject}.csv", rows, PROBE_FIELDS)
    print(f"[Saved] {out_dir / f'probe_summary_{args.subject}.csv'}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="EEGNet 表征的 run 身份线性探针（单次训练 + 全量特征）。"
    )
    parser.add_argument("--subject", nargs="+", default=["sub-01"], help="被试编号，可指定多个")
    parser.add_argument("--task", default="imagine", choices=("imagine", "reading"))
    parser.add_argument("--top-k", type=int, default=10, help="保留样本数最多的前 k 个句类")
    parser.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="试次表 CSV 目录")
    parser.add_argument("--pkl-root", type=Path, default=PKL_ROOT, help="preprocessed_pkl 根目录")
    parser.add_argument("--target-sfreq", type=float, default=500.0, help="目标采样率")
    parser.add_argument("--epochs", type=int, default=30, help="EEGNet 训练轮数（固定）")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--f1", type=int, default=8, help="EEGNet 第一层时间滤波器数")
    parser.add_argument("--d", type=int, default=2, help="EEGNet 深度乘数")
    parser.add_argument("--f2", type=int, default=16, help="EEGNet 可分离卷积输出通道数")
    parser.add_argument("--kernel-seconds", type=float, default=0.5, help="第一层时间核长度（秒）")
    parser.add_argument("--separable-seconds", type=float, default=0.125, help="可分离时间核长度（秒）")
    parser.add_argument("--dropout", type=float, default=0.25, help="第一个 dropout")
    parser.add_argument("--dropout2", type=float, default=0.5, help="第二个 dropout")
    parser.add_argument("--folds", type=int, default=5, help="线性探针的交叉验证折数")
    parser.add_argument("--split-seed", type=int, default=42, help="8:2 划分种子（与 train.py 一致）")
    parser.add_argument("--seed", type=int, default=42, help="训练与探针的随机种子")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"), help="训练设备"
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 工作进程数")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出根目录")
    parser.add_argument("--run-name", default=None, help="本次实验目录名，默认按时间命名")
    parser.add_argument("--quiet-steps", action="store_true", help="不逐 epoch 打印训练日志")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs 至少为 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Device] {device}")

    run_name = args.run_name or f"run_probe_{datetime.now():%Y%m%d_%H%M%S}"
    subjects = list(args.subject)
    all_rows: list[dict] = []

    for subject in subjects:
        args.subject = subject
        print(f"\n===== 被试 {subject} =====", flush=True)
        dataset = EEGNetDataset(
            subject=subject,
            task=args.task,
            top_k=args.top_k,
            target_sfreq=args.target_sfreq,
            csv_root=args.csv_root,
            pkl_root=args.pkl_root,
        )
        out_dir = args.out_dir / run_name / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        all_rows.extend(run_subject(args, dataset, out_dir, device))
        del dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = args.out_dir / run_name / "probe_all.csv"
    write_rows(summary_path, all_rows, PROBE_FIELDS)
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
