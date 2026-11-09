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
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        # Transformer blocks
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # 只返回 class tokens 部分
        return tokens[:, : self.num_classes_token, :]


# ============================================================
# 3. 完整双分支模型
# ============================================================
class TimeFreqDualBranchNet(nn.Module):
    def __init__(self, num_channels=122, time_len=1000, num_freq_bands=5,
                 num_classes_token=4, num_classes=40, dim=128, depth=2,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.time_len = time_len

        # ---- 时域分支 ----
        # (B, C, T) -> (B, C, dim)
        self.time_csa = ChannelSelfAttention(num_channels, time_len, dim=dim)
        # Transformer 模块
        self.time_transformer = DomainTransformer(
            num_tokens=dim,
            feat_dim=num_channels,
            dim=dim,
            num_classes_token=num_classes_token,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )

        # ---- 频域分支 ----
        # 频 PSD 得到 PSD 得到 (B, C, F) -> (B, C, dim)
        self.freq_csa = ChannelSelfAttention(num_channels, num_freq_bands, dim=dim)
        self.freq_transformer = DomainTransformer(
            num_tokens=dim,
            feat_dim=num_channels,
            dim=dim,
            num_classes_token=num_classes_token,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )

        # ---- 多头 class-token 融合模块 (3.2.3) ----
        fusion_in = 2 * num_classes_token * dim
        hidden1 = fusion_in * 2      # 第一层扩到输入的 2 倍
        hidden2 = fusion_in // 2     # 第二层减半
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x_time, x_freq_psd):
        """
        x_time     : (B, C, T)   滤波后的时域 EEG
        x_freq_psd : (B, C, F)   五个频段的 PSD 特征 (F=5)
        """
        # ---------- 时域分支 ----------
        t = self.time_csa(x_time)        # (B, C, dim)
        t = t.transpose(1, 2)            # (B, dim, C)
        z_t = self.time_transformer(t)   # (B, N, dim)

        # ---------- 频域分支 ----------
        f = self.freq_csa(x_freq_psd)    # (B, C, dim)
        f = f.transpose(1, 2)            # (B, dim, C)
        z_f = self.freq_transformer(f)   # (B, N, dim)

        # ---------- class-token 拼接融合 ----------
        B = z_t.size(0)
        fused = torch.cat([z_t.reshape(B, -1), z_f.reshape(B, -1)], dim=-1)  # (B, 2*N*dim)
        logits = self.classifier(fused)
        return logits


# ============================================================
# 4. 快速自测
# ============================================================
if __name__ == "__main__":
    B, C, T, F_bands = 2, 122, 1651, 5
    model = TimeFreqDualBranchNet(
        num_channels=C, time_len=T, num_freq_bands=F_bands,
        num_classes_token=4, num_classes=40, dim=128, depth=2, num_heads=8,
    )

    x_time = torch.randn(B, C, T)        # 时域输入
    x_freq = torch.randn(B, C, F_bands)  # 频域 PSD 输入

    logits = model(x_time, x_freq)
    print("logits shape:", logits.shape)          # 期望 (B, 40)
    print("num params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")