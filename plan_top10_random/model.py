import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础单元: 多尺度并行卷积层 (纯时间多尺度版本)
# ------------------------------------------------------------
# 改动: 所有卷积分支的 kernel 从 (k,k) 改为 (1,k)
#       - 电极轴只用 1x1, 避免"伪空间卷积"
#       - 时间轴保留 3 / 5 多尺度感受野
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
            nn.Conv2d(mid_ch, branch_ch, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        # 分支 3: 1x1(stride) -> (1,5)  [仅时间轴]
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1, stride=s, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, branch_ch, kernel_size=(1, 5), padding=(0, 2), bias=False),
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
# 改动: 用 stride 下采样替代外层池化, 两路都是可学习卷积
#       stride 只降时间 (E 已在 Stem 压缩到很小)
# ============================================================
class PMCNNBlock(nn.Module):
    """
    一个 PMCNN block:
        x -> MultiScaleConvLayer(stride) -> (+ 残差) -> BN -> ReLU
    stride: 时间轴下采样倍数 (1 或 2), 电极轴始终不下采样。
            下采样完全由卷积/池化的 stride 完成, 不再插入独立池化层;
            shortcut 用同样 stride 的 1x1 卷积, 两路形状与算子性质一致。
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

        # 相加之后统一归一化, 两路的尺度差异在这里被消除
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ms(x) + self.shortcut(x)
        out = self.act(self.bn_out(out))
        return out


# ============================================================
# 3. EEG Stem: 1 -> C0 时间卷积升维 + 电极可学习压缩
# ------------------------------------------------------------
# 改动: 新增电极压缩 (E_in -> E_out)
#   E_out = 1  -> 直接压成 "全局空间滤波器" (轻量)
#   E_out > 1  -> 学 E_out 个空间基向量 (保留少量空间结构)
# 使用 groups 控制参数量
# ============================================================
class EEGStem(nn.Module):
    """
    原始 EEG (B, 1, E_in, T) -> (B, C0, E_out, T)

    两步:
      1) 时间轴升维:   Conv2d(1, C0, (1, k))
      2) 电极轴压缩:   Conv2d(C0, C0, (E_in, 1), groups=C0)  (深度可分离)
                      然后得到 (B, C0, 1, T), 若 E_out>1 再线性变换到 E_out

    简化实现: 用一个 (E_in, 1) 深度卷积 + 1x1 卷积实现:
         (B, C0, E_in, T) --(E_in,1,groups=C0)--> (B, C0, 1, T)
         --(1x1, C0->C0*E_out) view --> (B, C0, E_out, T)
    实际更简洁的做法: 直接 (E_in,1) 普通卷积, 输出 C_out_spatial 个空间基,
    再沿通道重排为 E_out 个电极位置。
    这里用最简单直观的方式: (E_in,1) 卷积, 输出通道 C0, 然后 broadcast/expand
    到 E_out 个电极位置 (E_out 通过 repeat 或 interpolate 不重要),
    为通用性, 采用以下方案:

        Conv2d(C0, E_out*C0, kernel_size=(E_in,1), groups=C0)  -> 参数少
        view -> (B, C0, E_out, T)
    """
    def __init__(self, C0=64, k=7, E_in=122, E_out=16):
        super().__init__()
        assert E_out >= 1
        self.E_in = E_in
        self.E_out = E_out

        # 1) 时间升维
        self.temporal = nn.Sequential(
            nn.Conv2d(1, C0, kernel_size=(1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(C0),
            nn.ReLU(inplace=True),
        )

        # 2) 电极压缩: 深度卷积 (E_in,1) 把 E_in 压到 1, 输出 C0 通道
        self.spatial_depthwise = nn.Conv2d(
            C0, C0, kernel_size=(E_in, 1), groups=C0, bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(C0)
        self.spatial_act = nn.ReLU(inplace=True)

        # 3) 若 E_out > 1, 学 E_out 个空间基 (1x1 卷积: C0 -> E_out*C0, 分组)
        #    expand 后补 BN, 避免输出尺度未归一化直接送进 block1
        if E_out > 1:
            self.spatial_expand = nn.Sequential(
                nn.Conv2d(C0, E_out * C0, kernel_size=1, groups=C0, bias=False),
                nn.BatchNorm2d(E_out * C0),
            )
        else:
            self.spatial_expand = None

    def forward(self, x):
        # x: (B, 1, E_in, T)
        x = self.temporal(x)                     # (B, C0, E_in, T)

        # 电极压缩: (E_in,1) 深度卷积 -> (B, C0, 1, T)
        x = self.spatial_depthwise(x)
        x = self.spatial_act(self.spatial_bn(x))

        if self.spatial_expand is not None:
            # (B, E_out*C0, 1, T) -> (B, C0, E_out, T)
            B, _, _, T = x.shape
            x = self.spatial_expand(x)           # (B, E_out*C0, 1, T)
            x = x.view(B, self.E_out, -1, 1, T)  # (B, E_out, C0, 1, T)
            x = x.squeeze(3).permute(0, 2, 1, 3).contiguous()  # (B, C0, E_out, T)
        # 若 E_out==1, x 已经是 (B, C0, 1, T)
        return x                                 # (B, C0, E_out, T)


# ============================================================
# 4. 可选: 电极注意力池化 (E 维很小时锦上添花)
# ============================================================
class ElectrodeAttentionPool(nn.Module):
    """
    在 E 轴上做 attention pooling:
        输入: (B, C, E, T)
        输出: (B, C, 1, T)
    对每个 (B, C) 通道, 在 E 上算 softmax 权重
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
        # x: (B, C, E, T)
        B, C, E, T = x.shape
        # 对每个 (b, e, t) 用通道向量打分: (B, C, E, T) -> 展平 C
        feat = x.permute(0, 2, 3, 1).contiguous()        # (B, E, T, C)
        logits = self.score(feat).squeeze(-1)            # (B, E, T)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)   # (B, E, T, 1)
        out = (feat * w).sum(dim=1)                      # (B, T, C)
        out = out.permute(0, 2, 1).unsqueeze(2)          # (B, C, 1, T)
        return out


# ============================================================
# 5. 完整 PMCNN 网络
# ------------------------------------------------------------
# 改动: E 参数真正传给 Stem 用于电极压缩;
#       增加 E_out 与 use_electrode_attn 选项
# ============================================================
class PMCNN(nn.Module):
    """
    输入:  (B, 1, E_in, T)   原始 EEG (单通道)
    输出:  (B, num_classes)

    结构:
        Stem: 1 -> C0, E_in -> E_out
        Block1: C0 -> 128
        Block2: 128 -> 256, 时间下采样 x2
        Block3: 256 -> 256, 时间下采样 x2
        Block4: 256 -> 512, 时间下采样 x2
        [可选] 电极注意力池化
        GlobalAvgPool(T) -> Dropout -> FC
    """
    def __init__(self, num_classes=2, E_in=122, E_out=16,
                 C0=64, stem_k=7, dropout=0.5, use_electrode_attn=False):
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
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

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

    def forward(self, x):              # (B, 1, E_in, T)
        x = self.stem(x)               # (B, C0, E_out, T)
        x = self.block1(x)             # (B, 128, E_out, T)
        x = self.block2(x)             # (B, 256, E_out, T/2)
        x = self.block3(x)             # (B, 256, E_out, T/4)
        x = self.block4(x)             # (B, 512, E_out, T/8)

        if self.use_electrode_attn:
            x = self.electrode_pool(x) # (B, 512, 1, T/8)

        x = self.global_pool(x)        # (B, 512, 1, 1)
        x = torch.flatten(x, 1)        # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)                 # (B, num_classes)
        return x


# ============================================================
# 6. 快速自测: 打印每层 shape (E=122)
# ============================================================
if __name__ == "__main__":
    B, E, T = 8, 122, 1651
    num_classes = 39

    # 方案 1: E_out=16, 保留少量空间结构, 不用电极注意力
    model = PMCNN(num_classes=num_classes, E_in=E, E_out=16, C0=64, stem_k=7, use_electrode_attn=True)
    x = torch.randn(B, 1, E, T)

    print("=" * 60)
    print(f"输入: {tuple(x.shape)}")
    with torch.no_grad():
        y = model.stem(x);   print(f"stem   : {tuple(y.shape)}")
        y = model.block1(y); print(f"block1 : {tuple(y.shape)}")
        y = model.block2(y); print(f"block2 : {tuple(y.shape)}")
        y = model.block3(y); print(f"block3 : {tuple(y.shape)}")
        y = model.block4(y); print(f"block4 : {tuple(y.shape)}")
        y = model.global_pool(y); print(f"pool   : {tuple(y.shape)}")
        y = torch.flatten(y, 1); print(f"flatten: {tuple(y.shape)}")
        out = model.fc(y);   print(f"输出   : {tuple(out.shape)}")
    print("=" * 60)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[E_out=16]  可训练参数: {n_params:,} ({n_params/1e6:.2f} M)")

    # 方案 2: E_out=1, 纯时间网络 (最轻量)
    model2 = PMCNN(num_classes=num_classes, E_in=E, E_out=1,
                   C0=64, stem_k=7, use_electrode_attn=False)
    n2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    with torch.no_grad():
        out2 = model2(torch.randn(2, 1, E, T))
    print(f"[E_out=1 ]  可训练参数: {n2:,} ({n2/1e6:.2f} M), 输出: {tuple(out2.shape)}")

    # 方案 3: E_out=16 + 电极注意力
    model3 = PMCNN(num_classes=num_classes, E_in=E, E_out=16,
                   C0=64, stem_k=7, use_electrode_attn=True)
    n3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    with torch.no_grad():
        out3 = model3(torch.randn(2, 1, E, T))
    print(f"[E_out=16 + attn] 可训练参数: {n3:,} ({n3/1e6:.2f} M), 输出: {tuple(out3.shape)}")

    print("=" * 60)