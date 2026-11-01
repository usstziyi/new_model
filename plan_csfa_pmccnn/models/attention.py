import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力 (2.4.3节)"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # (B, C, H, W) -> (B, C, 1, 1)
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
    """空间注意力 (2.4.4节)"""
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


class FrequencyChannelAttention(nn.Module):
    """频率通道注意力 (2.4.5节) - 基于FFT频谱统计的通道加权"""
    def __init__(self, channels, reduction=16):
        super(FrequencyChannelAttention, self).__init__()
        # 采用类似通道注意力的门控结构作用于频谱
        # 区别只在于：上面个两种注意力输入的是空间域池化结果，这里的输入是频域幅度谱的池化结果。
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # 1. 二维快速傅里叶变换
        # rfft2 只计算沿最后一维的非负频率部分
        # 输入：(B, C, H, W) -> 输出：(B, C, H, W//2 + 1)
        fft_x = torch.fft.rfft2(x, norm='ortho')
        # 2. 取幅度谱
        amp = torch.abs(fft_x)
        # 3. 全局池化获取每个通道的频域统计量
        # 在频率维度（H 和 W/2+1 两个频率轴）上做全局平均，得到 (B, C)
        # 含义：每个通道的平均频谱能量
        pooled_amp = torch.mean(amp, dim=(2, 3))
        # 4. 生成频域权重
        weight = self.fc(pooled_amp).view(B, C, 1, 1)
        # 5. 权重作用于原图 (这里采用类似通道注意力的加权方式)
        return x * weight

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
        B, C, H, W = x.size()
        
        # 1. 傅里叶变换
        # rfft2 只计算沿最后一维的非负频率部分
        # 输入：(B, C, H, W) -> 输出：(B, C, H, W//2 + 1)
        fft_x = torch.fft.rfft2(x, norm='ortho') # (B, C, H, W//2 + 1)
        
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
        
        # 7. 逆傅里叶变换，回到时间/空间域
        out = torch.fft.irfft2(fft_weighted, s=(H, W), norm='ortho')
        
        return out