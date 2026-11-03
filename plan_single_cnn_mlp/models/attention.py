import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力 (2.4.3节)"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # (B, C, E, T) -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的两层感知机
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力 (2.4.4节)：在 (E, T) 平面上做二维卷积生成权重图"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 卷积生成权重图
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class FrequencyAttention(nn.Module):
    """真正的频率注意力 (2.4.5节) - 在频域进行幅度加权"""
    def __init__(self, channels, reduction=16):
        super(FrequencyAttention, self).__init__()
        # 这里的设计可以很灵活。为了简单，我们可以对幅度谱的通道做注意力
        # 或者对频率维度做一维卷积。这里采用通道注意力类似的门控
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, E, T = x.size()
        
        # 1. 一维傅里叶变换：只沿时间轴 T 做
        # 电极轴上的"空间频率"没有生理意义，因此不对 E 做 FFT
        # 输入：(B, C, E, T) -> 输出：(B, C, E, T//2 + 1)
        fft_x = torch.fft.rfft(x, dim=-1, norm='ortho') # (B, C, E, T//2 + 1)
        
        # 2. 分离幅度和相位
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 3. 对幅度谱进行全局池化提取特征 (B, C)
        pooled_amp = torch.mean(amp, dim=(2, 3))
        
        # 4. 生成权重 (B, C, 1, 1)
        weight = self.fc(pooled_amp).view(B, C, 1, 1)
        
        # 5. 对幅度谱进行加权！
        amp_weighted = amp * weight
        
        # 6. 重构复数频域信号 (利用极坐标转直角坐标)
        # 注意：这里需要用到 torch.polar 或者 cos/sin 组合
        real = amp_weighted * torch.cos(phase)
        imag = amp_weighted * torch.sin(phase)
        fft_weighted = torch.complex(real, imag)
        
        # 7. 逆一维傅里叶变换，回到时间域，用 n=T 保证还原出原始长度
        out = torch.fft.irfft(fft_weighted, n=T, dim=-1, norm='ortho')
        
        return out