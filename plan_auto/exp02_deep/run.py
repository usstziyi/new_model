"""实验 02：braindecode 标准模型 + 按 run 分组交叉验证 + 增强/TTA。

与既有 ``plan_braindecode_baseline`` 的关键区别：
1. 划分改为**按 run 分组**（旧实验是随机试次划分，会高估）。
2. 输入降到 125 Hz，并按试次逐通道 z-score + 截断，减少慢漂移影响。
3. 训练时做平移/缩放/噪声/通道丢弃/时间遮挡增强，测试时做平移 TTA。
4. 每折用训练 run 里留出的验证集选 epoch，测试折只用于最终评估。

输出：``plan_auto/exp02_deep/outputs/<model>/<subject>/``
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402
from plan_auto.common.utils import Tee, get_device, set_seed, timestamp  # noqa: E402
from plan_auto.exp02_deep import models as MODELS  # noqa: E402
from plan_auto.exp02_deep.augment import augment, tta_shifts  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "plan_auto/exp02_deep/outputs"


class EEGTrials(Dataset):
    """按试次返回 (C, T)；训练时在线增强。"""

    def __init__(self, eeg, indices, labels, train: bool = False, seed: int = 0, decimate: int = 1):
        self.eeg = eeg
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.decimate = decimate

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.asarray(self.eeg[idx], dtype=np.float32)
        if self.decimate > 1:
            x = np.ascontiguousarray(x[:, :: self.decimate])
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / np.maximum(std, 1e-6)
        np.clip(x, -10.0, 10.0, out=x)
        if self.train:
            x = augment(x, self.rng)
        return torch.from_numpy(x), int(self.labels[idx])


@torch.no_grad()
def predict(model, loader, device, tta: bool, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        batch = x.numpy()
        variants = tta_shifts(batch) if tta else [batch]
        batch_probs = None
        for variant in variants:
            tensor = torch.from_numpy(np.ascontiguousarray(variant)).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(tensor)
            p = torch.softmax(logits.float(), dim=1).cpu().numpy()
            batch_probs = p if batch_probs is None else batch_probs + p
        probs.append(batch_probs / len(variants))
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def evaluate(model, loader, device, tta: bool, amp: bool) -> dict:
    prob, y = predict(model, loader, device, tta, amp)
    pred = prob.argmax(axis=1)
    loss = float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)).mean())
    return {
        "loss": loss,
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "prob": prob,
    }


def run_fold(
    model_name: str,
    subject: str,
    fold: int,
    eeg,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    device,
    n_times: int,
) -> dict:
    set_seed(args.seed + fold)
    n_classes = int(labels.max()) + 1
    model = MODELS.build_model(model_name, eeg.shape[1], n_times, n_classes, args.sfreq).to(device)

    train_loader = DataLoader(
        EEGTrials(eeg, fit_idx, labels, train=args.aug, seed=args.seed + fold, decimate=args.decimate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        EEGTrials(eeg, val_idx, labels, decimate=args.decimate),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        EEGTrials(eeg, test_idx, labels, decimate=args.decimate),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    best_states: list[tuple[float, int, dict]] = []
    best_score, best_epoch, bad = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        total_loss, n_samples = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            n_samples += len(y)
        scheduler.step()

        val = evaluate(model, val_loader, device, tta=False, amp=amp)
        score = val["balanced_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_samples, 1),
                "val_loss": val["loss"],
                "val_acc": val["accuracy"],
                "val_balanced_acc": score,
                "val_macro_f1": val["macro_f1"],
                "seconds": time.time() - started,
            }
        )
        print(
            f"[{model_name}][{subject}][f{fold}] epoch {epoch:02d} "
            f"train={total_loss / max(n_samples, 1):.3f} val={val['accuracy']:.3f} "
            f"bacc={score:.3f} ({time.time() - started:.1f}s)",
            flush=True,
        )

        if score > best_score + 1e-4:
            best_score, best_epoch, bad = score, epoch, 0
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_states.append((score, epoch, state))
            best_states = sorted(best_states, key=lambda item: item[0], reverse=True)[:3]
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{model_name}][{subject}][f{fold}] 早停于 epoch {epoch}", flush=True)
                break

    # 用验证集 top-3 epoch 做快照集成，降低“挑到噪声 epoch”的方差
    probs = []
    for _, _, state in best_states:
        model.load_state_dict(state)
        probs.append(evaluate(model, test_loader, device, tta=args.tta, amp=amp)["prob"])
    prob = np.mean(probs, axis=0) if probs else evaluate(model, test_loader, device, tta=args.tta, amp=amp)["prob"]
    y_test = np.concatenate([y.numpy() for _, y in test_loader])
    pred = prob.argmax(axis=1)
    return {
        "prob": prob,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_score,
        "best_state": best_states[0][2] if best_states else None,
        "test_accuracy": float(np.mean(pred == y_test)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "n_train": len(fit_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }


def run_subject(model_name: str, subject: str, args, device) -> dict:
    cache = D.load_cache(subject, target_sfreq=250)
    eeg = np.asarray(cache["eeg"], dtype=np.float32)
    labels = np.asarray(cache["labels"])
    runs = np.asarray(cache["runs"])
    n_chars = np.asarray(cache["n_chars"])
    n_times = int(np.ceil(eeg.shape[-1] / args.decimate))

    matched = D.length_matched_indices(labels, n_chars, seed=args.seed)
    matched_mask = np.zeros(len(labels), dtype=bool)
    matched_mask[matched] = True

    folds = D.grouped_folds(runs, labels, n_splits=args.n_splits, seed=args.seed)
    n_classes = int(labels.max()) + 1
    y_pred = np.full(len(labels), -1, dtype=np.int64)
    prob = np.zeros((len(labels), n_classes), dtype=np.float32)

    out_dir = OUT_ROOT / model_name / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        fit_idx, val_idx = D.split_train_val(train_idx, runs, labels, val_ratio=0.2, seed=args.seed + 100 + fold)
        result = run_fold(
            model_name, subject, fold, eeg, labels, fit_idx, val_idx, test_idx, args, device, n_times
        )
        y_pred[test_idx] = result["prob"].argmax(axis=1)
        prob[test_idx] = result["prob"]
        pd.DataFrame(result["history"]).to_csv(out_dir / f"fold{fold}_history.csv", index=False)
        if result["best_state"] is not None:
            torch.save(result["best_state"], out_dir / f"fold{fold}.pt")
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": result["best_epoch"],
                "best_val_balanced_acc": result["best_val_balanced_acc"],
                "test_accuracy": result["test_accuracy"],
                "test_balanced_accuracy": result["test_balanced_accuracy"],
                "test_macro_f1": result["test_macro_f1"],
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_test": result["n_test"],
            }
        )
        print(
            f"[{model_name}][{subject}] fold {fold} best_epoch={result['best_epoch']} "
            f"test_acc={result['test_accuracy']:.3f} test_bacc={result['test_balanced_accuracy']:.3f}",
            flush=True,
        )

    if np.any(y_pred < 0):
        raise RuntimeError(f"{subject} 有未预测试次")
    result = M.classification_metrics(labels, y_pred, n_classes)
    _, low, high = M.cluster_bootstrap_ci(labels, y_pred, runs)
    result.update(
        {
            "model": model_name,
            "subject": subject,
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "matched_accuracy": float(np.mean(y_pred[matched_mask] == labels[matched_mask])),
            "matched_n": int(matched_mask.sum()),
            "binomial_p_full": M.binomial_p(int(np.sum(y_pred == labels)), len(labels), 1 / n_classes),
            "binomial_p_matched": M.binomial_p(
                int(np.sum(y_pred[matched_mask] == labels[matched_mask])),
                int(matched_mask.sum()),
                1 / n_classes,
            ),
            "n_trials": int(len(labels)),
            "n_params": int(sum(p.numel() for p in MODELS.build_model(
                model_name, eeg.shape[1], n_times, n_classes, args.sfreq).parameters())),
            "config": vars(args),
        }
    )
    pd.DataFrame(fold_rows).to_csv(out_dir / "folds.csv", index=False)
    np.savez_compressed(
        out_dir / "oof.npz",
        y_true=labels,
        y_pred=y_pred,
        prob=prob,
        runs=runs,
        n_chars=n_chars,
        matched_mask=matched_mask,
    )
    M.save_json(out_dir / "summary.json", result)
    print(
        f"[{model_name}][{subject}] OOF acc={result['accuracy']:.3f} "
        f"CI=[{low:.3f},{high:.3f}] matched={result['matched_accuracy']:.3f}",
        flush=True,
    )
    return result


def pool_results(model_name: str, subjects: list[str]) -> dict:
    y_true, y_pred, groups, matched = [], [], [], []
    for subject in subjects:
        data = np.load(OUT_ROOT / model_name / subject / "oof.npz")
        y_true.append(data["y_true"])
        y_pred.append(data["y_pred"])
        groups.append(np.char.add(f"{subject}:", data["runs"].astype(str)))
        matched.append(data["matched_mask"])
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    groups = np.concatenate(groups)
    matched = np.concatenate(matched)
    n_classes = int(y_true.max()) + 1
    pooled = M.classification_metrics(y_true, y_pred, n_classes)
    _, low, high = M.cluster_bootstrap_ci(y_true, y_pred, groups)
    pooled.update(
        {
            "model": model_name,
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "matched_accuracy": float(np.mean(y_pred[matched] == y_true[matched])),
            "matched_n": int(matched.sum()),
            "binomial_p_full": M.binomial_p(int(np.sum(y_pred == y_true)), len(y_true), 1 / n_classes),
            "binomial_p_matched": M.binomial_p(
                int(np.sum(y_pred[matched] == y_true[matched])), int(matched.sum()), 1 / n_classes
            ),
            "n_subjects": len(subjects),
        }
    )
    M.save_json(OUT_ROOT / model_name / "pooled_summary.json", pooled)
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description="实验02：braindecode 深度模型，按 run 分组 CV")
    parser.add_argument("--models", nargs="+", default=["eegnet"])
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sfreq", type=int, default=125)
    parser.add_argument("--decimate", type=int, default=2, help="从 250 Hz 缓存再降采样的倍数")
    parser.add_argument("--folds", type=int, default=5, dest="n_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = get_device()
    print(f"[Device] {device} | models={args.models} | subjects={args.subjects}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Tee(OUT_ROOT / f"exp02_{args.tag or 'run'}_{timestamp()}.log"):
        for model_name in args.models:
            results = []
            for subject in args.subjects:
                results.append(run_subject(model_name, subject, args, device))
            table = pd.DataFrame(results).sort_values("subject")
            table.to_csv(OUT_ROOT / model_name / "results.csv", index=False)
            pooled = pool_results(model_name, args.subjects)
            print(
                f"[Pooled][{model_name}] acc={pooled['accuracy']:.3f} "
                f"CI=[{pooled['accuracy_ci_low']:.3f},{pooled['accuracy_ci_high']:.3f}] "
                f"matched={pooled['matched_accuracy']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
