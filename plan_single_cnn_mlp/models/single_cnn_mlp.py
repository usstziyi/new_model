import torch
import torch.nn as nn

from .attention import ChannelAttention, SpatialAttention, FrequencyAttention
from .ms_conv import MultiScaleConvLayer


class SingleBranchCNNMLP(nn.Module):
    """单分支 EEG 语义分类模型（多尺度卷积 + 注意力 + GAP + MLP）。

    与 CSFA-PMCCNN 的区别：

    - 不再分 α / β 两条分支：各频段在通道维拼接成一个输入，只走一条卷积主干；
    - 去掉分支融合、LayerNorm、初级胶囊与动态路由；
    - 分类头换成「全局平均池化 -> 两层 MLP」，输出 logits，
      配合 ``nn.CrossEntropyLoss``。

    Args:
        in_channels: 输入通道数，即频段个数（α+β 时为 ``2``，单频段时为 ``1``）。
        num_classes: 语义类别数，ds005170 为 ``39``。
        hidden_dim: MLP 隐层维度，默认 ``128``。
        dropout: MLP 前的 dropout 概率，默认 ``0.5``。

    Input:
        ``(B, in_channels, E, T)``，``E = 122`` 为电极数，``T`` 为时间采样点数。

    Output:
        ``(B, num_classes)`` 未归一化的 logits。
    """

    def __init__(self, in_channels=2, num_classes=39, hidden_dim=128, dropout=0.5):
        super().__init__()

        # ================= 卷积主干（单分支） =================
        self.ms_conv1 = MultiScaleConvLayer(in_channels, 64)
        self.ca = ChannelAttention(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.ms_conv2 = MultiScaleConvLayer(64, 128)
        self.sa = SpatialAttention()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ms_conv3 = MultiScaleConvLayer(128, 256)
        self.fa = FrequencyAttention(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ================= GAP + MLP 分类头 =================
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, in_channels, E, T)
        x = self.pool1(self.ca(self.ms_conv1(x)))
        x = self.pool2(self.sa(self.ms_conv2(x)))
        x = self.pool3(self.fa(self.ms_conv3(x)))

        x = self.gap(x)             # (B, 256, 1, 1)
        return self.classifier(x)   # (B, num_classes)
