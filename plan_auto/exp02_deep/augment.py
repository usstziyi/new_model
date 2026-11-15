"""实验 02 的数据增强。

全部在标准化后的单试次信号上做，只用于训练；验证 / 测试不增强。
"""

from __future__ import annotations

import numpy as np


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x: (C, T) float32，返回同形状的增强副本。"""
    x = x.copy()
    _, n_times = x.shape

    # 时间平移（零填充），模拟想象起止的抖动
    shift = int(rng.integers(-25, 26))
    if shift != 0:
        pad = np.zeros_like(x)
        if shift > 0:
            pad[:, shift:] = x[:, :-shift]
        else:
            pad[:, :shift] = x[:, -shift:]
        x = pad

    # 幅度缩放
    x *= float(rng.uniform(0.9, 1.1))

    # 高斯噪声
    x += rng.normal(0.0, 0.05, size=x.shape).astype(np.float32)

    # 通道丢弃
    if rng.random() < 0.3:
        drop = rng.choice(x.shape[0], size=max(1, int(0.05 * x.shape[0])), replace=False)
        x[drop] = 0.0

    # 时间遮挡
    if rng.random() < 0.5:
        width = int(rng.integers(10, 40))
        start = int(rng.integers(0, max(n_times - width, 1)))
        x[:, start:start + width] = 0.0

    return x


def tta_shifts(x: np.ndarray, shifts=(-25, -12, 0, 12, 25)) -> list[np.ndarray]:
    """测试时平移增强，支持 (C, T) 或 (B, C, T)，沿最后一维平移。"""
    n_times = x.shape[-1]
    variants = []
    for shift in shifts:
        if shift == 0:
            variants.append(x)
            continue
        out = np.zeros_like(x)
        if shift > 0:
            out[..., shift:] = x[..., :-shift]
        else:
            out[..., :shift] = x[..., -shift:]
        variants.append(out)
    return variants
