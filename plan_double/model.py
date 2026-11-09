import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ============================================================
# 1. 通道自注意力模块
#    把每个通道当作一个 token，在通道维度做自注意力
#    输入(B, C, T/F)
#    输出(B, C, dim)
# ============================================================
class ChannelSelfAttention(nn.Module):
    def __init__(self, num_channels, seq_len, dim):
        super().__init__()
        self.w_q = nn.Linear(seq_len, dim)
        self.w_k = nn.Linear(seq_len, dim)
        self.w_v = nn.Linear(seq_len, dim)

    def forward(self, x):
        Q = self.w_q(x)              # (B, C, dim)
        K = self.w_k(x)              # (B, C, dim)
        V = self.w_v(x)              # (B, C, dim)

        # 缩放点积注意力 式(3.1)，内部等价于 softmax(QK^T / sqrt(d_q)) V
        out = F.scaled_dot_product_attention(Q, K, V)   # (B, C, dim)
        return out


# ============================================================
# 1.5 时域卷积 stem
#     逐通道做多尺度一维卷积，把 (B, C, T) 压成 (B, C, dim)
#     替代原来的 Linear(T -> dim)：保留时序局部性，参数从 634K 降到约 29K
# ============================================================
class ConvStem(nn.Module):
    def __init__(self, dim=128, kernels=(25, 50, 100), ch_per=8,
                 pool_bins=8, dropout=0.1):
        """
        kernels   : 各分支的时间卷积核长（500 Hz 下 25/50/100 点约 50/100/200 ms）
        ch_per    : 每个分支的输出通道数
        pool_bins : 时间轴分段平均的段数，保留粗粒度时序结构
        """
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1, ch_per, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(ch_per),
                nn.ELU(inplace=True),
            )
            for k in kernels
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        feat_dim = ch_per * len(kernels) * pool_bins
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, dim))

    def forward(self, x):
        """x: (B, C, T) -> (B, C, dim)"""
        B, C, T = x.shape
        # 把通道并进 batch 维，卷积核在各通道间共享
        h = x.reshape(B * C, 1, T)
        # 偶数核长（50/100）在 padding=k//2 时会多出一个点，裁掉保持时间轴对齐
        h = torch.cat([branch(h)[..., :T] for branch in self.branches], dim=1)
        h = self.pool(h).reshape(B * C, -1)
        return self.proj(h).reshape(B, C, -1)


# ============================================================
# 2. 时间域 / 频率域 Transformer 模块
#    两个域共用同一结构：class-token + 位置编码 + Transformer Encoder
#    Encoder Block 直接使用官方 nn.TransformerEncoderLayer（Pre-LN + GELU）
# ============================================================
class DomainTransformer(nn.Module):
    def __init__(self, num_tokens, feat_dim, dim=128, num_classes_token=4,
                 depth=2, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        """
        num_tokens   : 该域的 token 数（= 通道自注意力输出维度）
        feat_dim     : 每个 token 的特征维度（= 通道数 C）
        dim          : Transformer 隐层维度，须能被 num_heads 整除
        num_classes_token : 可训练 class token 数 N
        mlp_ratio    : FFN 隐层扩张倍数
        """
        super().__init__()
        self.dim = dim
        self.num_classes_token = num_classes_token

        # 通道数 C（如 122）通常不能被 num_heads 整除，先投影到 dim
        self.input_proj = nn.Linear(feat_dim, dim)

        # 可训练 class tokens（截断正态初始化，std=0.02，与 ViT 一致）
        self.class_tokens = nn.Parameter(torch.empty(1, num_classes_token, dim))
        trunc_normal_(self.class_tokens, std=0.02)

        # 可学习位置编码，截断正态初始化
        self.pos_embed = nn.Parameter(torch.empty(1, num_classes_token + num_tokens, dim))
        trunc_normal_(self.pos_embed, std=0.02)

        # 官方 Encoder 层：norm_first=True 即 Pre-LN，与原实现一致
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, num_tokens, feat_dim)
        返回: class token 特征 (B, N, dim)
        """
        B = x.size(0)
        tokens = self.input_proj(x)      # (B, num_tokens, dim)

        # 拼接 class tokens -> (B, N + num_tokens, dim)
        cls = self.class_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # 位置编码
        tokens = tokens + self.pos_embed

        # Transformer blocks
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # 只返回 class tokens 部分
        return tokens[:, : self.num_classes_token, :]


# ============================================================
# 3. 完整双分支模型
# ============================================================
class TimeFreqDualBranchNet(nn.Module):
    def __init__(self, num_channels=122, num_freq_bands=5,
                 num_classes_token=4, num_classes=40, dim=128, depth=2,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_freq_bands = num_freq_bands

        # PSD 各 (通道, 频段) 的量级差异很大（γ 约 -3.1、δ 约 -1.1 log 单位），
        # 不标准化时线性层会被少数大尺度维度主导，实测连训练集都拟合不动
        self.psd_norm = nn.BatchNorm1d(num_channels * num_freq_bands)

        # ---- 时域分支 ----
        # (B, C, T) -> (B, C, dim)，多尺度一维卷积保留时序局部性
        self.time_stem = ConvStem(dim=dim, dropout=dropout)
        # Transformer 模块
        self.time_transformer = DomainTransformer(
            num_tokens=dim,
            feat_dim=num_channels,
            dim=dim,
            num_classes_token=num_classes_token,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )

        # ---- 频域分支 ----
        # (B, C, F) -> (B, C, dim)，把每个通道的 F 个频段当特征做通道自注意力
        self.freq_csa = ChannelSelfAttention(num_channels, num_freq_bands, dim=dim)
        self.freq_transformer = DomainTransformer(
            num_tokens=dim,
            feat_dim=num_channels,
            dim=dim,
            num_classes_token=num_classes_token,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )

        # ---- 多头 class-token 融合模块 (3.2.3) ----
        # 论文是四层 MLP（先 2 倍再减半），但那是大数据集；这里训练样本仅 1952 条，
        # 原写法 1024->2048->512->10 有 3.15M 参数（占全模型 68%），实测会造成
        # 类别坍缩，故压缩为两层、隐层取输入的 1/4。
        fusion_in = 2 * num_classes_token * dim
        hidden = max(fusion_in // 4, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x_time, x_freq_psd):
        """
        x_time     : (B, C, T)   滤波后的时域 EEG
        x_freq_psd : (B, C, F)   五个频段的 PSD 特征 (F=5)
        """
        B = x_freq_psd.size(0)
        # PSD 逐 (通道, 频段) 标准化
        psd = self.psd_norm(x_freq_psd.reshape(B, -1)).reshape(
            B, self.num_channels, self.num_freq_bands
        )

        # ---------- 时域分支 ----------
        t = self.time_stem(x_time)       # (B, C, dim)
        t = t.transpose(1, 2)            # (B, dim, C)
        z_t = self.time_transformer(t)   # (B, N, dim)

        # ---------- 频域分支 ----------
        f = self.freq_csa(psd)           # (B, C, dim)
        f = f.transpose(1, 2)            # (B, dim, C)
        z_f = self.freq_transformer(f)   # (B, N, dim)

        # ---------- class-token 拼接融合 ----------
        fused = torch.cat([z_t.reshape(B, -1), z_f.reshape(B, -1)], dim=-1)
        return self.classifier(fused)


# ============================================================
# 4. 快速自测
# ============================================================
if __name__ == "__main__":
    B, C, T, F_bands = 2, 122, 1651, 5
    model = TimeFreqDualBranchNet(
        num_channels=C, num_freq_bands=F_bands,
        num_classes_token=4, num_classes=40, dim=128, depth=2, num_heads=8,
    )

    x_time = torch.randn(B, C, T)        # 时域输入
    x_freq = torch.randn(B, C, F_bands)  # 频域 PSD 输入

    logits = model(x_time, x_freq)
    print("logits shape:", logits.shape)          # 期望 (B, 40)
    print("num params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")