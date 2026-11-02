"""在留出 run 上评估训练好的 CSFA-PMCCNN checkpoint。

默认使用 checkpoint 里记录的 task / subjects / target_sfreq / normalize，
在 day 5（run 37-45）上评估，也可以手动指定天数与被试。

用法::

    uv run plan_csfa_pmccnn/test.py --checkpoint outputs/csfa_pmccnn/<run>/best.pt

    # 换到 day 4 验证集评估
    uv run plan_csfa_pmccnn/test.py --checkpoint outputs/csfa_pmccnn/<run>/best.pt --days 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import load_class_names
from models import CSFAPMCCNN
from train import (
    RUNS_PER_DAY,
    MarginLoss,
    build_parts,
    concat_or_single,
    make_eval_loader,
    report_test,
    runs_from_days,
)


def parse_args():
    parser = argparse.ArgumentParser(description="评估 CSFA-PMCCNN EEG 语义分类 checkpoint。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt 路径")
    parser.add_argument("--days", nargs="+", type=int, default=[5], help="用第几天作为测试集")
    parser.add_argument("--runs-per-day", type=int, default=RUNS_PER_DAY, help="每天 run 数")
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
    target_sfreq = checkpoint.get("target_sfreq", 500.0)
    normalize = checkpoint.get("normalize", True)
    num_classes = checkpoint["num_classes"]
    print(
        f"[Checkpoint] task={task} subjects={subjects} epoch={checkpoint.get('epoch')} "
        f"sfreq={target_sfreq:g} normalize={normalize} classes={num_classes}"
    )

    runs = runs_from_days(args.days, args.runs_per_day)
    parts = build_parts(subjects, task, runs, target_sfreq, normalize, args.cache_size)
    dataset = concat_or_single(parts)
    loader = make_eval_loader(
        dataset, args.batch_size, args.num_workers, device.type == "cuda"
    )
    print(f"[Split] days={args.days} runs={runs[0]}-{runs[-1]} samples={len(dataset)}")

    model = CSFAPMCCNN(in_channels=checkpoint["in_channels"], num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    class_names = load_class_names()
    out_dir = args.out_dir or args.checkpoint.parent
    metrics = report_test(
        model,
        loader,
        MarginLoss(),
        device,
        class_names,
        out_dir,
        args.max_batches,
    )
    print(f"[Done] acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
