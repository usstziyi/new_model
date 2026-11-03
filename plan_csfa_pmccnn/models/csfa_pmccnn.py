import torch
import torch.nn as nn
from .ms_conv import MultiScaleConvLayer
from .attention import ChannelAttention, SpatialAttention, FrequencyAttention
from .capsule import PrimaryCaps, DigitCaps

class CSFAPMCCNN(nn.Module):
    """CSFA-PMCCNN：双分支（α / β）多尺度卷积 + 注意力 + 胶囊分类。

    真实 EEG 输入:
        x_alpha / x_beta: ``(B, C, E, T)``，``C = 1``（特征通道维），
        ``E = 122``（电极数），``T`` 为时间采样点数（随任务而定）。
    输出:
        ``(B, num_classes)``，每个类别的胶囊模长，可直接交给 Margin Loss。

    空间结构:
        融合后的特征图 ``(B, 256, E', T')`` 经自适应池化压缩到固定的
        ``(grid_size, grid_size)`` 网格（默认 3×3），保留局部时空信息，
        不做全局池化。初级胶囊数 = ``out_capsules * grid_size**2``
        （默认 ``8 * 3 * 3 = 72``），因此 ``DigitCaps`` 的变换矩阵
        ``W`` 形状为 ``(1, 72, num_classes, 16, 16)``；``out_capsules=16``
        时初级胶囊数为 144，``W`` 形状为 ``(1, 144, num_classes, 16, 16)``。

    Args:
        in_channels: 输入特征通道数，真实 EEG 用 ``1``。
        num_classes: 语义类别数，ds005170 为 ``39``。
        out_capsules: 每个空间位置的初级胶囊数，默认 ``8``（3×3 网格时共 72 个）。
        grid_size: 空间网格边长，默认 ``3``（即 3×3 = 9 个空间位置）。
    """

    def __init__(self, in_channels=1, num_classes=39, out_capsules=8, grid_size=3):
        super(CSFAPMCCNN, self).__init__()
        
        # ================= 上分支 (α) =================
        self.alpha_ms_conv1 = MultiScaleConvLayer(in_channels, 64)
        self.alpha_ca = ChannelAttention(64)
        self.alpha_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.alpha_ms_conv2 = MultiScaleConvLayer(64, 128)
        self.alpha_sa = SpatialAttention()
        self.alpha_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.alpha_ms_conv3 = MultiScaleConvLayer(128, 256)
        self.alpha_fa = FrequencyAttention(256)
        self.alpha_pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # ================= 下分支 (β) =================
        self.beta_ms_conv1 = MultiScaleConvLayer(in_channels, 64)
        self.beta_ca = ChannelAttention(64)
        self.beta_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.beta_ms_conv2 = MultiScaleConvLayer(64, 128)
        self.beta_sa = SpatialAttention()
        self.beta_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.beta_ms_conv3 = MultiScaleConvLayer(128, 256)
        self.beta_fa = FrequencyAttention(256)
        self.beta_pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # ================= 后端分类 =================
        # 不做全局池化，而是把特征图自适应压缩到 grid_size × grid_size 的固定网格，
        # 每个空间位置都保留下来参与胶囊投票（默认 3×3 = 9 个位置）。
        self.grid_size = grid_size
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        # 用 1×1 卷积代替原来的全连接层：通道维 256 -> 128，逐空间位置独立作用
        self.fc1 = nn.Conv2d(256, 128, kernel_size=1)
        # 真实 EEG 经三级卷积 / 注意力 / 池化后，融合特征幅度很低（约 1e-2）。
        # 若直接送入初级胶囊，两层 squash 会把范数压到 1e-10，Margin Loss 失去梯度。
        # 这里先做 LayerNorm，把每个位置的特征标准化到 O(1)。
        self.fc_norm = nn.LayerNorm(128)

        # 初级胶囊：每个空间位置 out_capsules 个 16 维胶囊
        # -> 共 out_capsules * grid_size**2 个（默认 8 * 3 * 3 = 72）
        self.out_capsules = out_capsules
        self.primary_caps = PrimaryCaps(
            in_channels=128, 
            out_capsules=out_capsules, 
            capsule_dim=16, 
            kernel_size=1, 
            stride=1,
        )
        # in_caps 与 PrimaryCaps 的输出保持同源，避免两处手工不同步
        self.digit_caps = DigitCaps(
            in_caps=out_capsules * grid_size ** 2,
            in_dim=16,
            num_caps=num_classes,
            out_dim=16,
            num_routing=3,
        )
        
    def forward(self, x_alpha, x_beta):
        # x_alpha / x_beta: (B, C, E, T)，E 为电极数，T 为时间采样点数
        # ---------------- 第一阶段 ----------------
        a = self.alpha_ca(self.alpha_ms_conv1(x_alpha))
        a = self.alpha_pool1(a)
        b = self.beta_ca(self.beta_ms_conv1(x_beta))
        b = self.beta_pool1(b)
        
        # ---------------- 第二阶段 ----------------
        a = self.alpha_sa(self.alpha_ms_conv2(a))
        a = self.alpha_pool2(a)
        b = self.beta_sa(self.beta_ms_conv2(b))
        b = self.beta_pool2(b)
        
        # ---------------- 第三阶段 ----------------
        a = self.alpha_fa(self.alpha_ms_conv3(a))
        a = self.alpha_pool3(a)
        b = self.beta_fa(self.beta_ms_conv3(b))
        b = self.beta_pool3(b)
        
        # ---------------- 融合与后端 ----------------
        fused = a + b                                  # (B, 256, E', T')
        out = self.spatial_pool(fused)                 # (B, 256, g, g)
        out = torch.relu(self.fc1(out))                # (B, 128, g, g)
        # LayerNorm 需要把特征维放到最后： (B, g, g, 128)，逐空间位置归一化
        out = self.fc_norm(out.permute(0, 2, 3, 1))    # (B, g, g, 128)
        out = out.permute(0, 3, 1, 2).contiguous()     # (B, 128, g, g)

        # ---------------- 胶囊层处理 ----------------
        # 初级胶囊提取： (B, 128, g, g) -> (B, 8*g*g, 16)，g = grid_size
        primary_out = self.primary_caps(out)
        
        # 分类胶囊输出
        digit_out = self.digit_caps(primary_out) # 形状 (B, num_classes, 16)
        
        # 计算胶囊模长作为最终概率 (论文公式 3.2)
        probs = torch.norm(digit_out, dim=-1) 
        
        return probs