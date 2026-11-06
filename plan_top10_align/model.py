"""PMCNN + 768 维投影头，用于 EEG-文本嵌入空间对齐。

与 ``plan_top10_random/model.py`` 的区别：

- 主干（Stem + 4 个 PMCNNBlock）完全保留；
- 去掉分类输出，改为 ``512 -> proj_hidden -> embed_dim`` 的投影头，输出 L2 归一化
  的 768 维向量（与冻结 LaBSE 的句向量空间对齐）；
- ``fc`` 保留，仅用于可选的辅助交叉熵（``--ce-weight > 0`` 时才参与训练）；
- 投影头单独用 xavier 初始化，避免沿用小 std 初始化导致对齐收敛慢。

输入:  (B, 1, E_in, T)   原始 EEG（单通道）
输出:  (B, embed_dim)    已 L2 归一化的 EEG 向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础单元: 多尺度并行卷积层 (纯时间多尺度版本)
# ------------------------------------------------------------
# 所有卷积分支的 kernel 从 (k,k) 改为 (1,k)
#   - 电极轴只用 1x1, 避免"伪空间卷积"
#   - 时间轴保留 3 / 5 多尺度感受野
# 输入:  (B, C_in, E, T)
# 输出:  (B, C_out, E, T)
# ============================================================
class MultiScaleConvLayer(nn.Module):
    """
    并行多尺度卷积层 (时间轴多尺度, 电极轴 1x1)

    stride 只作用于时间轴的 1x1 卷积与 AvgPool (stride=(1, stride)),
    时间卷积 (1,3)/(1,5) 始终 stride=1:
    四个分支在同一位置降采样, 时间轴严格对齐, 且时间卷积在 T/stride
    上计算, 比"先全分辨率卷积再池化"更省算力。

    4 分支各输出 C_out//4 通道, 沿 dim=1 拼接:
      分支1: 1x1 Conv (stride)
      分支2: 1x1 Conv (stride) -> (1,3) Conv
      分支3: 1x1 Conv (stride) -> (1,5) Conv
      分支4: (1,3) AvgPool (stride) -> 1x1 Conv
    """

    def __init__(self, in_channels, out_channels, reduction=4, stride=1):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels 必须能被 4 整除"
        branch_ch = out_channels // 4
        mid_ch = max(in_channels // reduction, 8)
        s = (1, stride)

        # 分支 1: 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # 分支 2: 1x1(stride) -> (1,3)  [仅时间轴]
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, branch_ch, kernel_size=(1, 3),
                      padding=(0, 1), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # 分支 3: 1x1(stride) -> (1,5)  [仅时间轴]
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, branch_ch, kernel_size=(1, 5),
                      padding=(0, 2), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # 分支 4: (1,3) AvgPool(stride) -> 1x1  [仅时间轴]
        # k=3,p=1 与 1x1 stride 卷积的输出长度公式一致:
        #   floor((T-1)/stride)+1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 3), stride=s, padding=(0, 1)),
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ============================================================
# 2. PMCNN Block: MultiScaleConv + 残差 (+ 可选下采样)
# ------------------------------------------------------------
class PMCNNBlock(nn.Module):
    """
    一个 PMCNN block:
        x -> MultiScaleConvLayer(stride) -> (+ 残差) -> BN -> ReLU
    stride: 时间轴下采样倍数 (1 或 2), 电极轴始终不下采样。
    """

    def __init__(self, in_channels, out_channels, stride=1, reduction=4):
        super().__init__()
        self.ms = MultiScaleConvLayer(in_channels, out_channels, reduction, stride)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=(1, stride), bias=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ms(x) + self.shortcut(x)
        out = self.act(self.bn_out(out))
        return out


# ============================================================
# 3. EEG Stem: 1 -> C0 时间卷积升维 + 真空间卷积
# ------------------------------------------------------------
class EEGStem(nn.Module):
    """
    原始 EEG (B, 1, E_in, T) -> (B, C0, E_out, T)

    三步:
      1) 时间轴升维:    Conv2d(1, C0, (1, k))
      2) 真空间卷积:    Conv2d(C0, E_out, (E_in, 1), groups=1)
      3) 逐电极特征升维: Conv2d(E_out, E_out*C0, 1, groups=E_out)
    """

    def __init__(self, C0=64, k=7, E_in=122, E_out=16):
        super().__init__()
        assert E_out >= 1
        self.C0 = C0
        self.E_in = E_in
        self.E_out = E_out

        # 1) 时间升维
        self.temporal = nn.Sequential(
            nn.Conv2d(1, C0, kernel_size=(1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(C0),
            nn.ReLU(inplace=True),
        )

        # 2) 真空间卷积: kernel 跨全部电极, groups=1 让通道之间也参与混合
        self.spatial = nn.Conv2d(
            C0, E_out, kernel_size=(E_in, 1), bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(E_out)
        self.spatial_act = nn.ReLU(inplace=True)

        # 3) 每个虚拟电极升维到 C0 个特征 (groups=E_out: 电极之间独立)
        self.feature_expand = nn.Sequential(
            nn.Conv2d(E_out, E_out * C0, kernel_size=1, groups=E_out, bias=False),
            nn.BatchNorm2d(E_out * C0),
        )

    def forward(self, x):
        # x: (B, 1, E_in, T)
        x = self.temporal(x)                     # (B, C0, E_in, T)

        # 真空间卷积: 每个输出通道是一个覆盖全部电极的空间滤波器
        x = self.spatial_act(self.spatial_bn(self.spatial(x)))  # (B, E_out, 1, T)

        # 逐电极升维: 输出通道序号 k = e*C0 + c, 与下面 reshape 的分组一致
        B, _, _, T = x.shape
        x = self.feature_expand(x)               # (B, E_out*C0, 1, T)
        x = x.reshape(B, self.E_out, self.C0, 1, T).squeeze(3)
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, C0, E_out, T)
        return x


# ============================================================
# 4. 可选: 电极注意力池化
# ============================================================
class ElectrodeAttentionPool(nn.Module):
    """
    在 E 轴上做 attention pooling:
        输入: (B, C, E, T)
        输出: (B, C, 1, T)
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or max(channels // 4, 8)
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        B, C, E, T = x.shape
        feat = x.permute(0, 2, 3, 1).contiguous()        # (B, E, T, C)
        logits = self.score(feat).squeeze(-1)            # (B, E, T)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)   # (B, E, T, 1)
        out = (feat * w).sum(dim=1)                      # (B, T, C)
        out = out.permute(0, 2, 1).unsqueeze(2)          # (B, C, 1, T)
        return out


# ============================================================
# 5. 对齐用 PMCNN: 主干 + 投影头
# ============================================================
class PMCNN(nn.Module):
    """
    输入:  (B, 1, E_in, T)   原始 EEG (单通道)
    输出:  (B, embed_dim)    L2 归一化的 EEG 向量

    结构:
        Stem: 1 -> C0, E_in -> E_out
        Block1: C0 -> 128
        Block2: 128 -> 256, 时间下采样 x2
        Block3: 256 -> 256, 时间下采样 x2
        Block4: 256 -> 512, 时间下采样 x2
        [可选] 电极注意力池化
        GlobalAvgPool -> Dropout -> 512 -> 768 投影 -> L2 归一化

    额外接口:
        features(x)   -> (B, 512)  池化后的时空特征
        embed(feats)  -> (B, embed_dim)  归一化向量
        classify(feats) -> (B, num_classes)  仅辅助 CE 使用
    """

    def __init__(self, num_classes=10, E_in=122, E_out=16,
                 C0=64, stem_k=7, dropout=0.5, use_electrode_attn=False,
                 embed_dim=768, proj_hidden=512):
        super().__init__()

        self.stem = EEGStem(C0=C0, k=stem_k, E_in=E_in, E_out=E_out)

        self.block1 = PMCNNBlock(C0, 128)
        self.block2 = PMCNNBlock(128, 256, stride=2)
        self.block3 = PMCNNBlock(256, 256, stride=2)
        self.block4 = PMCNNBlock(256, 512, stride=2)

        self.use_electrode_attn = use_electrode_attn
        if use_electrode_attn:
            self.electrode_pool = ElectrodeAttentionPool(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)   # (B, 512, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(512, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, embed_dim),
        )
        # 仅用于可选的辅助交叉熵（--ce-weight > 0）
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()
        # 投影头单独初始化：主干的 std=0.01 对对齐任务偏小
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def features(self, x):             # (B, 1, E_in, T)
        x = self.stem(x)               # (B, C0, E_out, T)
        x = self.block1(x)             # (B, 128, E_out, T)
        x = self.block2(x)             # (B, 256, E_out, T/2)
        x = self.block3(x)             # (B, 256, E_out, T/4)
        x = self.block4(x)             # (B, 512, E_out, T/8)

        if self.use_electrode_attn:
            x = self.electrode_pool(x) # (B, 512, 1, T/8)

        x = self.global_pool(x)        # (B, 512, 1, 1)
        return torch.flatten(x, 1)     # (B, 512)

    def embed(self, features):
        z = self.projection(self.dropout(features))
        return F.normalize(z, p=2, dim=-1)

    def classify(self, features):
        return self.fc(features)

    def forward(self, x):
        return self.embed(self.features(x))


# ============================================================
# 6. 快速自测: 打印每层 shape
# ============================================================
if __name__ == "__main__":
    B, E, T = 4, 122, 1651
    model = PMCNN(num_classes=10, E_in=E, E_out=16, C0=64, stem_k=7,
                  use_electrode_attn=True, embed_dim=768)
    x = torch.randn(B, 1, E, T)

    with torch.no_grad():
        feats = model.features(x)
        z = model.embed(feats)
        logits = model.classify(feats)

    print("=" * 60)
    print(f"输入     : {tuple(x.shape)}")
    print(f"features : {tuple(feats.shape)}")
    print(f"embed    : {tuple(z.shape)} | L2 范数 {z.norm(dim=-1).tolist()}")
    print(f"logits   : {tuple(logits.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_params:,} ({n_params/1e6:.2f} M)")
    print("=" * 60)
