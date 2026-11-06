"""评估 top-10 类随机划分实验的 checkpoint。

从 checkpoint 读取被试、采样率、归一化、通道配置与**划分参数**，用同一
``split_seed`` 复现 train / val / test 划分，默认在 test 上评估。

用法::

    uv run plan_top10_random/test.py --checkpoint outputs/top10_random/<run>/best.pt

    # 在验证集上评估
    uv run plan_top10_random/test.py --checkpoint outputs/top10_random/<run>/best.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import TOP_ROOT, TopClassEEGDataset, split_indices
from model import PMCNN
from train import report_test


def parse_args():
    parser = argparse.ArgumentParser(description="评估 top-10 类 PMCNN checkpoint。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt 路径")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="评估哪个划分")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--max-batches", type=int, default=None, help="最多评估批数（试跑用）")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认 checkpoint 所在目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    subject = checkpoint["subject"]
    target_sfreq = checkpoint.get("target_sfreq", 500.0)
    normalize = checkpoint.get("normalize", True)
    use_64_channels = checkpoint.get("use_64_channels", False)
    split_args = checkpoint.get("split", {"val_ratio": 0.1, "test_ratio": 0.1, "seed": 42})
    model_kwargs = checkpoint["model_kwargs"]
    class_names = dict(enumerate(checkpoint["class_names"]))
    print(
        f"[Checkpoint] subject={subject} epoch={checkpoint.get('epoch')} "
        f"sfreq={target_sfreq:g} normalize={normalize} "
        f"channels={model_kwargs['E_in']} classes={model_kwargs['num_classes']} "
        f"split={split_args}"
    )

    dataset = TopClassEEGDataset(
        subject=subject,
        root=args.data_root,
        target_sfreq=target_sfreq,
        normalize=normalize,
        use_64_channels=use_64_channels,
    )
    # 标签重映射由 dataset 按 type_number 升序现场生成，必须与训练时一致，
    # 否则模型输出下标会静默对应到错误的类别名。
    saved_ids = checkpoint.get("class_ids")
    if saved_ids is not None and [int(i) for i in saved_ids] != list(dataset.class_ids):
        raise RuntimeError(
            f"类别映射不一致: checkpoint={list(saved_ids)} dataset={list(dataset.class_ids)}"
        )

    train_idx, val_idx, test_idx = split_indices(
        dataset.labels,
        split_args["val_ratio"],
        split_args["test_ratio"],
        split_args["seed"],
    )
    index_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    subset = Subset(dataset, index_map[args.split])
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[Split] {args.split} samples={len(subset)}")

    model = PMCNN(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    out_dir = args.out_dir or args.checkpoint.parent
    metrics = report_test(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        class_names,
        out_dir,
        args.max_batches,
    )
    print(f"[Done] acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
