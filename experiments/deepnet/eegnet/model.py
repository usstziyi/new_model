"""EEGNet 时域基线（PyTorch 实现）。

参考：

    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact
    convolutional neural network for EEG-based brain-computer interfaces[J].
    Journal of Neural Engineering, 2018, 15(5): 056013.

实现要点：

- 输入为原始时域波形 ``(N, 1, C, T)``；
- 第一层时间卷积、深度可分离空间卷积、可分离时间卷积；
- **时间核长度按秒定义**，再按实际采样率换算。原论文在 128 Hz 下使用
  ``kernel_length=64``（约 0.5 s）与 ``separable_kernel=16``（约 0.125 s）；
  本实现默认也取 0.5 s / 0.125 s，因此在 500 Hz、250 Hz 下感受野与原版一致，
  而不是把 64 个采样点直接搬到 500 Hz。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kernel_length_samples(seconds: float, sfreq: float) -> int:
    """把秒级核长换算成采样点，并取最近的奇数。

    取奇数是为了让 ``padding="same"`` 在奇偶长度下都严格保持时间长度，
    避免 PyTorch 对偶数核的零填充告警。
    """
    length = max(3, int(round(seconds * float(sfreq))))
    return length if length % 2 == 1 else length + 1


class EEGNet(nn.Module):
    """紧凑型时域 EEG 分类网络。

    Args:
        n_channels: EEG 通道数。
        n_times: 每个试次的时间点数（重采样后的长度）。
        n_classes: 句类数。
        sfreq: 实际采样率，用于把秒级核长换算成采样点。
        F1: 第一层时间滤波器个数。
        D: 深度乘数，空间卷积输出通道为 ``F1 * D``。
        F2: 可分离卷积输出通道数。
        kernel_seconds: 第一层时间核长度（秒），默认 0.5 s。
        separable_seconds: 可分离时间核长度（秒），默认 0.125 s。
        dropout: 第一个池化后的 dropout。
        dropout2: 第二个池化后的 dropout。
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        sfreq: float = 500.0,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_seconds: float = 0.5,
        separable_seconds: float = 0.125,
        dropout: float = 0.25,
        dropout2: float = 0.5,
    ):
        super().__init__()
        if n_channels < 1 or n_times < 1:
            raise ValueError(f"n_channels/n_times 需为正数，收到 {n_channels}/{n_times}")

        kernel_length = kernel_length_samples(kernel_seconds, sfreq)
        separable_kernel = kernel_length_samples(separable_seconds, sfreq)
        pooled_times = n_times // 4 // 8
        if pooled_times < 1:
            raise ValueError(
                f"n_times={n_times} 太短：两次池化（4×8）后没有剩余时间点，"
                "请提高 target_sfreq 或延长试次"
            )

        self.n_channels = n_channels
        self.n_times = n_times
        self.sfreq = float(sfreq)
        self.kernel_length = kernel_length
        self.separable_kernel = separable_kernel
        self.n_features = F2 * pooled_times

        # 1. 时间卷积：每个通道独立，学习一组时间滤波器
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding="same", bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # 2. 深度空间卷积：在每个时间滤波器上跨全部电极做空间加权
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # 3. 可分离时间卷积：逐通道时间卷积 + 1×1 点卷积
        self.sep_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, separable_kernel),
            padding="same",
            groups=F1 * D,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout2)

        # 4. 分类头
        self.classifier = nn.Linear(self.n_features, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类头之前的展平特征 ``(N, n_features)``。

        供线性探针使用：同一个前向过程，只是在 ``self.classifier`` 之前截断。
        """
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = self.sep_depthwise(x)
        x = F.elu(self.bn3(self.sep_pointwise(x)))
        x = self.drop2(self.pool2(x))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, 1, C, T)``，返回 ``(N, n_classes)`` 的 logits。"""
        return self.classifier(self.embed(x))


def build_model(
    n_channels: int,
    n_times: int,
    n_classes: int,
    sfreq: float = 500.0,
    f1: int = 8,
    d: int = 2,
    f2: int = 16,
    kernel_seconds: float = 0.5,
    separable_seconds: float = 0.125,
    dropout: float = 0.25,
    dropout2: float = 0.5,
) -> EEGNet:
    """构造 EEGNet，参数命名与命令行保持一致。"""
    return EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        sfreq=sfreq,
        F1=f1,
        D=d,
        F2=f2,
        kernel_seconds=kernel_seconds,
        separable_seconds=separable_seconds,
        dropout=dropout,
        dropout2=dropout2,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
