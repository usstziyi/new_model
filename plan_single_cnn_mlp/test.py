"""在训练时留出的验证集（默认 20% 的 run）上评估单分支 CNN + MLP checkpoint。

评估使用 checkpoint 里记录的 task / subjects / bands / target_sfreq / normalize
以及 train.py 划出的 ``val_runs``，因此与训练时的验证集完全一致。

用法::

    uv run plan_single_cnn_mlp/test.py --checkpoint outputs/single_cnn_mlp/<run>/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from dataset import DEFAULT_BANDS, load_class_names
from models import SingleBranchCNNMLP
from train import build_parts, concat_or_single, make_eval_loader, report_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="评估单分支 CNN + MLP EEG 语义分类 checkpoint。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt 路径")
    parser.add_argument("--subjects", nargs="+", default=None, help="覆盖 checkpoint 里的被试")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--cache-size", type=int, default=1, help="每个被试缓存的 run 数")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--max-batches", type=int, default=None, help="最多评估批数（试跑用）")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认 checkpoint 所在目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task = checkpoint["task"]
    subjects = args.subjects or checkpoint["subjects"]
    bands = tuple(checkpoint.get("bands", DEFAULT_BANDS))
    target_sfreq = checkpoint.get("target_sfreq", 500.0)
    normalize = checkpoint.get("normalize", True)
    num_classes = checkpoint["num_classes"]
    val_runs = checkpoint.get("val_runs")
    if not val_runs:
        raise RuntimeError("checkpoint 中没有 val_runs，请用当前版 train.py 重新训练后再评估")
    print(
        f"[Checkpoint] task={task} subjects={subjects} bands={bands} "
        f"epoch={checkpoint.get('epoch')} sfreq={target_sfreq:g} "
        f"normalize={normalize} classes={num_classes}"
    )

    parts = build_parts(subjects, task, val_runs, bands, target_sfreq, normalize, args.cache_size)
    dataset = concat_or_single(parts)
    loader = make_eval_loader(
        dataset, args.batch_size, args.num_workers, device.type == "cuda"
    )
    print(f"[Split] val_runs={val_runs} samples={len(dataset)}")

    model = SingleBranchCNNMLP(
        in_channels=checkpoint.get("in_channels", len(bands)),
        num_classes=num_classes,
        embed_dim=checkpoint.get("embed_dim", 768),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    class_names = load_class_names()
    out_dir = args.out_dir or args.checkpoint.parent
    metrics = report_metrics(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        class_names,
        out_dir,
        split_name="val",
        max_batches=args.max_batches,
    )
    print(f"[Done] acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
