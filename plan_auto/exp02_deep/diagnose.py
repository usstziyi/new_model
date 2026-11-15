"""诊断：EEGNet 能否在小样本上过拟合（验证训练管线本身是否正常）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common.utils import get_device, set_seed  # noqa: E402
from plan_auto.exp02_deep import models as MODELS  # noqa: E402
from plan_auto.exp02_deep.run import EEGTrials  # noqa: E402


def main() -> None:
    set_seed(0)
    device = get_device()
    cache = D.load_cache("sub-01")
    eeg = np.asarray(cache["eeg"], dtype=np.float32)[:, :, ::2]
    labels = np.asarray(cache["labels"])
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), 40, replace=False) for c in range(10)])
    rng.shuffle(idx)
    train_idx, test_idx = idx[:300], idx[300:]
    loader = DataLoader(
        EEGTrials(eeg, train_idx, labels, train=False), batch_size=32, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(EEGTrials(eeg, test_idx, labels), batch_size=64, shuffle=False)

    for amp in (False, True):
        model = MODELS.build_model("eegnet", eeg.shape[1], eeg.shape[2], 10, 125.0).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
        for epoch in range(1, 41):
            model.train()
            total, n = 0.0, 0
            for x, y in loader:
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
                total += float(loss.detach()) * len(y)
                n += len(y)
            if epoch % 5 == 0 or epoch == 1:
                model.eval()
                correct = 0
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                            pred = model(x).argmax(1)
                        correct += int((pred == y).sum())
                print(f"amp={amp} epoch={epoch:02d} loss={total / n:.3f} train_acc={correct / n:.3f}", flush=True)


if __name__ == "__main__":
    main()
