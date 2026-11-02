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

    Args:
        in_channels: 输入特征通道数，真实 EEG 用 ``1``。
        num_classes: 语义类别数，ds005170 为 ``39``。
    """

    def __init__(self, in_channels=1, num_classes=39):
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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        # 真实 EEG 经三级卷积 / 注意力 / 池化后，融合特征幅度很低（约 1e-2）。
        # 若直接送入初级胶囊，两层 squash 会把范数压到 1e-10，Margin Loss 失去梯度。
        # 这里先做 LayerNorm，把每个样本的特征标准化到 O(1)。
        self.fc_norm = nn.LayerNorm(128)
        
        # 胶囊层 (假设把 128 维特征作为 8 个 16 维的胶囊输入)
        self.primary_caps = PrimaryCaps(in_channels=128, out_capsules=8, capsule_dim=16, kernel_size=1, stride=1)
        # 这里的 in_caps=8, in_dim=16 对应上面的输出
        self.digit_caps = DigitCaps(in_caps=8, in_dim=16, num_caps=num_classes, out_dim=16, num_routing=3)
        
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
        
        # ---------------- 融合与全连接 ----------------
        fused = a + b
        out = self.global_pool(fused)
        out = torch.flatten(out, 1)
        out = self.fc_norm(torch.relu(self.fc1(out)))
        
        # ---------------- 胶囊层处理 ----------------
        # 此时 out 形状为 (B, 128)，为了输入初级胶囊(需要4D张量)，转成 (B, 128, 1, 1)
        out = out.view(out.size(0), -1, 1, 1)
        
        # 初级胶囊提取
        primary_out = self.primary_caps(out) # 形状 (B, 8, 16)
        
        # 分类胶囊输出
        digit_out = self.digit_caps(primary_out) # 形状 (B, num_classes, 16)
        
        # 计算胶囊模长作为最终概率 (论文公式 3.2)
        probs = torch.norm(digit_out, dim=-1) 
        
        return probs