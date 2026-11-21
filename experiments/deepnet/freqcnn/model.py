"""频域 CNN 基线（PyTorch 实现）。

输入为 ``(N, C, F, T_frames)`` 的 STFT log 幅度谱：C 个 EEG 通道作为
Conv2d 的输入通道，频率与时间作为两个空间轴。网络用三层 3×3 卷积 +
最大池化 + 全局平均池化，最后接线性分类头。

这是一个刻意保持简单的频域对照，用来回答：

- 在**相同的频域表示**下，把线性 LDA 换成 CNN 后，随机划分相对 LORO/LODO
  的虚高是否仍然存在；
- 因此它不应和时域 EEGNet 混为一谈：EEGNet 换的是表示，频域 CNN 主要换的是
  模型容量。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FreqCNN(nn.Module):
    """基于 STFT log 幅度谱的二维 CNN。

    Args:
        n_channels: EEG 通道数，作为 Conv2d 的输入通道。
        n_classes: 句类数。
        width: 第一层卷积通道数，后两层分别为 ``2×width``、``4×width``。
        dropout: 分类头前的 dropout。
        n_freqs: 频率轴长度（仅用于校验）。
        n_frames: 时间轴长度（仅用于校验）。
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        width: int = 32,
        dropout: float = 0.5,
        n_freqs: int | None = None,
        n_frames: int | None = None,
    ):
        super().__init__()
        if n_channels < 1:
            raise ValueError(f"n_channels 需为正数，收到 {n_channels}")
        if n_freqs is not None and n_freqs < 4:
            raise ValueError(f"频率轴太短（{n_freqs}），两次 2×2 池化后无法继续")
        if n_frames is not None and n_frames < 4:
            raise ValueError(f"时间轴太短（{n_frames}），两次 2×2 池化后无法继续")

        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.n_frames = n_frames

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, C, F, T)``，返回 ``(N, n_classes)`` 的 logits。"""
        return self.classifier(self.features(x))


def build_model(
    n_channels: int,
    n_classes: int,
    width: int = 32,
    dropout: float = 0.5,
    n_freqs: int | None = None,
    n_frames: int | None = None,
) -> FreqCNN:
    """构造频域 CNN，参数命名与命令行保持一致。"""
    return FreqCNN(
        n_channels=n_channels,
        n_classes=n_classes,
        width=width,
        dropout=dropout,
        n_freqs=n_freqs,
        n_frames=n_frames,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
