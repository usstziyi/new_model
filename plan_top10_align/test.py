"""评估 top-10 类 EEG-文本对齐实验的 checkpoint。

从 checkpoint 读取被试、采样率、归一化、通道配置、划分参数与对齐配置，用同一
``split_seed`` 复现 train / val / test 划分，输出：

- 划分内检索：R@1/5/10、MRR、median rank、检索 CE loss；
- 全量文本库检索（2459 候选）；
- kNN 诊断（用训练集 EEG 向量做 gallery）。

用法::

    uv run plan_top10_align/test.py --checkpoint outputs/top10_align/<run>/best.pt

    # 在验证集上评估
    uv run plan_top10_align/test.py --checkpoint outputs/top10_align/<run>/best.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import TOP_ROOT, TopClassEEGDataset, split_indices
from model import PMCNN
from text_encoder import LaBSETextEncoder
from train import extract_embeddings, report_split


def parse_args():
    parser = argparse.ArgumentParser(description="评估 top-10 类 EEG-文本对齐 checkpoint。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt 路径")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="评估哪个划分")
    parser.add_argument("--data-root", type=Path, default=TOP_ROOT, help="top10_imagine 根目录")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--max-batches", type=int, default=None, help="最多评估批数（试跑用）")
    parser.add_argument("--knn-k", type=int, default=5, help="kNN 诊断的近邻数")
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
    temperature = checkpoint.get("temperature", 0.07)
    text_model = checkpoint.get("text_model", "sentence-transformers/LaBSE")
    print(
        f"[Checkpoint] subject={subject} epoch={checkpoint.get('epoch')} "
        f"select={checkpoint.get('select_metric')}={checkpoint.get('score')} "
        f"sfreq={target_sfreq:g} normalize={normalize} "
        f"channels={model_kwargs['E_in']} classes={model_kwargs['num_classes']} "
        f"embed_dim={model_kwargs.get('embed_dim')} split={split_args}"
    )

    dataset = TopClassEEGDataset(
        subject=subject,
        root=args.data_root,
        target_sfreq=target_sfreq,
        normalize=normalize,
        use_64_channels=use_64_channels,
    )
    # 标签重映射由 dataset 按 type_number 升序现场生成，必须与训练时一致，
    # 否则诊断用的 kNN 标签会静默错位。
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

    target_subset = Subset(dataset, index_map[args.split])
    loader = DataLoader(
        target_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[Split] {args.split} samples={len(target_subset)}")

    print(f"[Text] 正在用 {text_model} 编码 {len(dataset.texts)} 个句子 ...", flush=True)
    text_encoder = LaBSETextEncoder(text_model)
    text_bank = text_encoder(dataset.texts).cpu()   # 评估统一在 CPU 上算

    model = PMCNN(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # kNN 诊断需要训练集 gallery；评估 train 划分时用自己当 gallery 没有意义
    train_gallery = None
    if args.split != "train":
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        z_train, _, y_train = extract_embeddings(model, train_loader, device)
        train_gallery = (z_train, y_train)
    else:
        print("[Note] --split train 时跳过 kNN 诊断（gallery 与评估集相同，指标无意义）")

    out_dir = args.out_dir or args.checkpoint.parent
    report_split(
        model, loader, text_bank, out_dir, args.split,
        temperature, device, model_kwargs["num_classes"],
        args.max_batches, None, train_gallery, args.knn_k,
    )


if __name__ == "__main__":
    main()
