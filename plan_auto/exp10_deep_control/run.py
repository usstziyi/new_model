"""实验 10：深度模型 + 位置去趋势（最终混杂控制）。

输入：1.5 Hz 高通缓存；对每个 run、每个通道，用时域多项式（5 阶）拟合
“电压 ~ run 内归一化位置”，减掉拟合值。这样同一 run 内相邻类别块之间的
慢变化/台阶被移除，只保留“同一位置附近试次之间”的差异。

如果 ShallowFBCSP 在这种输入上仍然只在机会水平，就可以排除“线性特征没抓到、
但深度模型能抓到的非线性语义结构”。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402
from plan_auto.common.utils import Tee, get_device, set_seed, timestamp  # noqa: E402
from plan_auto.exp09_confound.run import normalized_position, run_balanced_recall  # noqa: E402

HIGHPASS = 1.5
DETREND_ROOT = PROJECT_ROOT / "plan_auto/cache/detrended_hp15"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp10_deep_control/outputs"


def build_detrended(subject: str, degree: int = 5, overwrite: bool = False) -> Path:
    out = DETREND_ROOT / f"{subject}.npy"
    if out.exists() and not overwrite:
        return out
    D.build_cache(subject, highpass=HIGHPASS)
    cache = D.load_cache(subject, highpass=HIGHPASS)
    eeg = np.asarray(cache["eeg"], dtype=np.float32)
    runs = np.asarray(cache["runs"])
    pos = normalized_position(runs)
    result = np.empty_like(eeg, dtype=np.float16)
    for run in np.unique(runs):
        idx = np.flatnonzero(runs == run)
        p = pos[idx]
        design = np.vstack([p ** d for d in range(degree + 1)]).T
        block = eeg[idx].reshape(len(idx), -1)
        coef, *_ = np.linalg.lstsq(design, block, rcond=None)
        residual = block - design @ coef
        result[idx] = residual.reshape(len(idx), eeg.shape[1], eeg.shape[2]).astype(np.float16)
        print(f"[Detrend] {subject} run={run} n={len(idx)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, result)
    print(f"[Detrend] {subject} saved {out}", flush=True)
    return out


class Trials(Dataset):
    def __init__(self, data, indices, labels, decimate=2, augment=True, seed=0):
        self.data = data
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.decimate = decimate
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.data[idx], dtype=np.float32)[:, :: self.decimate]
        x = (x - x.mean(-1, keepdims=True)) / np.maximum(x.std(-1, keepdims=True), 1e-6)
        np.clip(x, -10, 10, out=x)
        if self.augment:
            shift = int(self.rng.integers(-25, 26))
            if shift:
                y = np.zeros_like(x)
                if shift > 0:
                    y[:, shift:] = x[:, :-shift]
                else:
                    y[:, :shift] = x[:, -shift:]
                x = y
            x = x * float(self.rng.uniform(0.9, 1.1))
            x = x + self.rng.normal(0, 0.05, size=x.shape).astype(np.float32)
            if self.rng.random() < 0.3:
                drop = self.rng.choice(x.shape[0], max(1, int(0.05 * x.shape[0])), replace=False)
                x[drop] = 0
            if self.rng.random() < 0.5:
                w = int(self.rng.integers(10, 40))
                s = int(self.rng.integers(0, max(x.shape[-1] - w, 1)))
                x[:, s:s + w] = 0
        return torch.from_numpy(x), int(self.labels[idx])


@torch.no_grad()
def predict(model, loader, device, tta=True):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        batch = x.numpy()
        variants = [batch]
        if tta:
            for shift in (-25, -12, 12, 25):
                out = np.zeros_like(batch)
                if shift > 0:
                    out[..., shift:] = batch[..., :-shift]
                else:
                    out[..., :shift] = batch[..., -shift:]
                variants.append(out)
        acc = None
        for variant in variants:
            t = torch.from_numpy(np.ascontiguousarray(variant)).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                p = torch.softmax(model(t).float(), 1).cpu().numpy()
            acc = p if acc is None else acc + p
        probs.append(acc / len(variants))
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train_fold(model, train_loader, val_loader, test_loader, device, epochs, lr, patience):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        sched.step()
        prob, y = predict(model, val_loader, device, tta=False)
        score = balanced_accuracy_score(y, prob.argmax(1))
        print(f"  epoch {epoch:02d} val_bacc={score:.3f} ({time.time() - started:.1f}s)", flush=True)
        if score > best_score + 1e-4:
            best_score, bad = score, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(model, test_loader, device, tta=True)[0], best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(D.SUBJECTS))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, dest="n_splits")
    args = parser.parse_args()

    from braindecode.models import ShallowFBCSPNet

    device = get_device()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with Tee(OUT_ROOT / f"exp10_{timestamp()}.log"):
        rows = []
        for subject in args.subjects:
            path = build_detrended(subject)
            data = np.load(path, mmap_mode="r")
            cache = D.load_cache(subject, highpass=HIGHPASS)
            labels, runs = cache["labels"], cache["runs"]
            n_times = int(np.ceil(data.shape[-1] / 2))
            folds = D.grouped_folds(runs, labels, args.n_splits, args.seed)
            matched = D.length_matched_indices(labels, cache["n_chars"], seed=args.seed)
            mask = np.zeros(len(labels), dtype=bool)
            mask[matched] = True
            pred = np.full(len(labels), -1)
            prob = np.zeros((len(labels), 10), dtype=np.float32)

            for fold, (tr, te) in enumerate(folds):
                fit_idx, val_idx = D.split_train_val(tr, runs, labels, 0.2, args.seed + fold)
                set_seed(args.seed + fold)
                model = ShallowFBCSPNet(
                    n_chans=data.shape[1], n_times=n_times, n_outputs=10, sfreq=125.0
                ).to(device)
                loaders = [
                    DataLoader(Trials(data, idx, labels, augment=aug, seed=args.seed + fold),
                               batch_size=args.batch_size, shuffle=shuffle, pin_memory=True,
                               drop_last=shuffle)
                    for idx, aug, shuffle in ((fit_idx, True, True), (val_idx, False, False), (te, False, False))
                ]
                p, best = train_fold(model, *loaders, device, args.epochs, args.lr, args.patience)
                pred[te] = p.argmax(1)
                prob[te] = p
                print(f"[{subject}] fold {fold} test_acc={np.mean(pred[te] == labels[te]):.3f}", flush=True)

            acc = float(np.mean(pred == labels))
            rb = run_balanced_recall(labels, pred, runs)
            row = {
                "subject": subject,
                "accuracy": acc,
                "run_balanced_recall": rb,
                "matched_accuracy": float(np.mean(pred[mask] == labels[mask])),
                "chance": 0.1,
            }
            rows.append(row)
            print(f"[{subject}] acc={acc:.3f} runbal={rb:.3f} matched={row['matched_accuracy']:.3f}", flush=True)
            np.savez_compressed(OUT_ROOT / f"{subject}_oof.npz", y_true=labels, y_pred=pred,
                                prob=prob, runs=runs, matched_mask=mask)
        import pandas as pd

        pd.DataFrame(rows).to_csv(OUT_ROOT / "results.csv", index=False)


if __name__ == "__main__":
    main()
