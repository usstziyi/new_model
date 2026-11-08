"""TFP-CTNet：第 4 章重建实现（非作者原码）。

参考：田亮《基于言语想象脑电信号的分类算法研究》第 4 章，式 (4.1)-(4.15)，图 4.1-4.4。

固定输入 (B, 122, 1651) -> (B, 7) 原始 logits。

通道顺序、电极坐标与拓扑邻接属于数据侧，由 ``dataset.py`` 的
``load_channel_layout`` / ``build_adjacency`` 提供；本文件只接收已建好的
``adjacency``，不读任何文件。

论文明确支持的部分：
- 通道拓扑感知输入与残差融合：X_topo = φ(X) + E_topo。
- 三个卷积尺度 (1x1, 3x3, 5x5)，每支 3 个 Conv-LN-GELU，逐元素相加融合。
- 卷积 token 嵌入 + 固定位置编码。
- 6 层 pre-LN Transformer 编码块（式 4.7-4.8）、全局平均池化、门控残差 MLP。

重建假设（论文未规定）：
- φ(X) 用 depthwise 时间 Conv1d(k=7) + 自适应池化到 topo_dim。
- E_topo = 行归一化 A @ 可学习通道嵌入。
- token 划分用 AdaptiveAvgPool2d((8,8)) 后按"空间优先"展平。
- 式 (4.10) 的 ρ(h) 用可学习 MLP 实现，未加显式频谱算子。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TFPCTNetConfig:
    n_times: int = 1651
    n_classes: int = 7
    topo_dim: int = 128
    conv_channels: int = 16
    spatial_tokens: int = 8
    temporal_tokens: int = 8
    d_model: int = 128
    n_heads: int = 8              # 论文"8"的解读：注意力头数
    n_layers: int = 6
    ff_mult: int = 4
    conv_dropout: float = 0.2
    transformer_dropout: float = 0.1
    mlp_dropout: float = 0.2


class TopologyEmbedding(nn.Module):
    """式 (4.2)：X_topo = φ(X) + E_topo，输出 (B, C, D_topo)。"""

    def __init__(self, n_chans: int, topo_dim: int, adjacency: Tensor):
        super().__init__()
        self.topo_dim = topo_dim
        self.temporal_conv = nn.Conv1d(
            n_chans, n_chans, kernel_size=7, padding=3, groups=n_chans, bias=False,
        )
        self.feature_ln = nn.LayerNorm(topo_dim)
        self.channel_embedding = nn.Parameter(torch.empty(n_chans, topo_dim))
        nn.init.normal_(self.channel_embedding, std=0.02)
        self.topo_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("adjacency", adjacency)

    def forward(self, x: Tensor) -> Tensor:
        phi_x = self.temporal_conv(x)                        # (B, C, T) 通道不交互
        phi_x = F.adaptive_avg_pool1d(phi_x, self.topo_dim)  # (B, C, D_topo)
        phi_x = self.feature_ln(phi_x)
        topo = self.adjacency @ self.channel_embedding       # (C, D_topo)
        return phi_x + self.topo_scale * topo.unsqueeze(0)


class ChannelLayerNorm2d(nn.Module):
    """按特征通道做 LayerNorm，输入输出均为 NCHW。"""

    def __init__(self, n_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvLNGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            ChannelLayerNorm2d(out_channels),
            nn.GELU(),
        )


class MultiScaleConvBlock(nn.Module):
    """图 4.2：3 个尺度分支 × 3 层 Conv-LN-GELU，逐元素相加（式 4.4）。"""

    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                ConvLNGELU(1, hidden_channels, kernel),
                ConvLNGELU(hidden_channels, hidden_channels, kernel),
                ConvLNGELU(hidden_channels, hidden_channels, kernel),
            )
            for kernel in (1, 3, 5)
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, D_topo)，二维轴分别是电极与池化后的时间
        x = x.unsqueeze(1)                                  # (B, 1, C, D_topo)
        return self.dropout(sum(branch(x) for branch in self.branches))


class ConvTokenEmbedding(nn.Module):
    """把卷积特征图池化并展平为 N = spatial_tokens × temporal_tokens 个 token。"""

    def __init__(self, conv_channels: int, d_model: int, spatial: int, temporal: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((spatial, temporal))
        self.project = nn.Linear(conv_channels, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x).flatten(start_dim=2).transpose(1, 2)  # (B, S*F, H)
        return self.project(x)                                # (B, N, d_model)


def sinusoidal_position_encoding(seq_len: int, d_model: int) -> Tensor:
    """固定正弦位置编码 (1, N, d)。"""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class PreNormTransformerBlock(nn.Module):
    """式 (4.7)-(4.8)：pre-LN 的多头自注意力 + 前馈网络。"""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z: Tensor) -> Tensor:
        normed = self.ln1(z)
        attn, _ = self.attn(normed, normed, normed, need_weights=False)
        z = z + self.drop1(attn)
        return z + self.ffn(self.ln2(z))


class GatedResidualMLP(nn.Module):
    """式 (4.10)-(4.15)：h' = h + sigmoid(ρ(h)) · sigmoid(g(h)) · u(h)。"""

    def __init__(self, d_model: int, n_classes: int, dropout: float):
        super().__init__()
        self.band_map = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, h: Tensor) -> Tensor:
        alpha_f = torch.sigmoid(self.band_map(h))
        alpha_g = torch.sigmoid(self.feature_gate(h))
        h_prime = h + alpha_g * self.value_mlp(h) * alpha_f
        return self.classifier(h_prime)                      # 原始 logits，不含 softmax


class TFPCTNet(nn.Module):
    """面向 (122, 1651) 言语想象 EEG 的第 4 章 TFP-CTNet 重建。

    Args:
        config: 结构超参；为 ``None`` 时用默认配置。
        adjacency: ``(C, C)`` 行归一化电极邻接，由 ``dataset.build_adjacency``
            生成；通道数 C 即模型输入通道数。
    """

    def __init__(self, config: Optional[TFPCTNetConfig] = None, *, adjacency: Tensor):
        super().__init__()
        cfg = TFPCTNetConfig() if config is None else config
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"adjacency 必须是方阵，实际 {tuple(adjacency.shape)}")
        self.config = cfg
        self.n_chans = adjacency.shape[0]

        self.topology = TopologyEmbedding(self.n_chans, cfg.topo_dim, adjacency)
        self.multiscale_conv = MultiScaleConvBlock(cfg.conv_channels, cfg.conv_dropout)
        self.token_embedding = ConvTokenEmbedding(
            cfg.conv_channels, cfg.d_model, cfg.spatial_tokens, cfg.temporal_tokens,
        )
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(
                cfg.spatial_tokens * cfg.temporal_tokens, cfg.d_model,
            ),
            persistent=False,
        )
        self.transformer = nn.Sequential(*[
            PreNormTransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.transformer_dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.head = GatedResidualMLP(cfg.d_model, cfg.n_classes, cfg.mlp_dropout)

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_chans, cfg.n_times):
            raise ValueError(
                f"期望 (B,{self.n_chans},{cfg.n_times})，实际 {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise TypeError("EEG 必须是浮点张量")
        x = self.topology(x)                          # (B, C, D_topo)
        x = self.multiscale_conv(x)                   # (B, H, C, D_topo)
        z = self.token_embedding(x) + self.position_encoding   # (B, N, d_model)
        z = self.transformer(z)                       # (B, N, d_model)
        return self.head(z.mean(dim=1))               # (B, n_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="只跑通 TFP-CTNet 的前向/反向。")
    parser.add_argument("--channels", type=int, default=122)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.channels < 1 or args.batch_size < 1:
        parser.error("--channels / --batch-size 必须为正整数")

    torch.manual_seed(0)
    # 冒烟测试用单位阵占位；真实拓扑从 dataset.build_adjacency 传入
    adjacency = torch.eye(args.channels)
    model = TFPCTNet(adjacency=adjacency)
    print(f"channels: {model.n_chans} (identity adjacency placeholder)")

    x = torch.randn(args.batch_size, model.n_chans, model.config.n_times)
    y = torch.randint(model.config.n_classes, (args.batch_size,))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    print(f"logits: {tuple(logits.shape)}  loss: {loss.item():.4f}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    print("forward/backward: PASS")


if __name__ == "__main__":
    main()
