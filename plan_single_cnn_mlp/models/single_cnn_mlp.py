import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ChannelAttention, SpatialAttention, FrequencyAttention
from .ms_conv import MultiScaleConvLayer


class SingleBranchCNNMLP(nn.Module):
    """单分支 EEG 语义分类模型（多尺度卷积 + 注意力 + GAP + MLP）。

    与 CSFA-PMCCNN 的区别：

    - 不再分 α / β 两条分支：各频段在通道维拼接成一个输入，只走一条卷积主干；
    - 去掉分支融合、LayerNorm、初级胶囊与动态路由；
    - 分类头换成「全局平均池化 -> 两层 MLP」，输出 logits，
      配合 ``nn.CrossEntropyLoss``；
    - 额外有一个 768 维线性投影头，把 GAP 特征映射到与 LaBSE 相同的向量空间
      并做 L2 归一化，用于 EEG-文本对比对齐。

    Args:
        in_channels: 输入通道数，即频段个数（α+β 时为 ``2``，单频段时为 ``1``）。
        num_classes: 语义类别数，ds005170 为 ``39``。
        hidden_dim: MLP 隐层维度，默认 ``128``。
        dropout: MLP 前的 dropout 概率，默认 ``0.5``。
        embed_dim: 对比对齐空间的维度，与文本编码器输出一致，LaBSE 为 ``768``。

    Input:
        ``(B, in_channels, E, T)``，``E = 122`` 为电极数，``T`` 为时间采样点数。

    Output:
        ``forward``: ``(B, num_classes)`` 未归一化的 logits。
        ``forward_with_embedding``: ``(logits, z)``，``z`` 为 ``(B, embed_dim)``
        已 L2 归一化的 EEG 向量。
    """

    def __init__(
        self,
        in_channels=2,
        num_classes=39,
        hidden_dim=128,
        dropout=0.5,
        embed_dim=768,
    ):
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

        # ============== EEG -> 文本空间投影头 ==============
        self.embed_dim = embed_dim
        self.projection = nn.Linear(256, embed_dim)

    def features(self, x):
        """返回 GAP 后的 256 维时空特征 ``(B, 256)``。"""
        x = self.pool1(self.ca(self.ms_conv1(x)))
        x = self.pool2(self.sa(self.ms_conv2(x)))
        x = self.pool3(self.fa(self.ms_conv3(x)))
        return self.gap(x).flatten(1)

    def forward(self, x):
        """分类 logits ``(B, num_classes)``。"""
        return self.classifier(self.features(x))

    def forward_with_embedding(self, x):
        """同时返回分类 logits 与 L2 归一化的对齐向量。"""
        features = self.features(x)
        logits = self.classifier(features)
        embedding = F.normalize(self.projection(features), p=2, dim=-1)
        return logits, embedding

