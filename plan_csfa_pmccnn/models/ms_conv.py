import torch
import torch.nn as nn

class MultiScaleConvLayer(nn.Module):
    """
    图 3.2 中的并行多尺度卷积层 (PMCNN 的基础单元)

    输入:  (B, C_in, H, W)
    输出:  (B, C_out, H, W)          # H/W 不变，只改变通道数

    4 个分支各输出 C_out // 4 个通道，最后沿 dim=1 拼接：
      分支1: 1x1 Conv                        (B, C_in, H, W) -> (B, C_out//4, H, W)
      分支2: 1x1 Conv -> 3x3 Conv(pad=1)     (B, C_in, H, W) -> (B, mid, H, W) -> (B, C_out//4, H, W)
      分支3: 1x1 Conv -> 5x5 Conv(pad=2)     (B, C_in, H, W) -> (B, mid, H, W) -> (B, C_out//4, H, W)
      分支4: 3x3 AvgPool(pad=1) -> 1x1 Conv  (B, C_in, H, W) -> (B, C_in, H, W) -> (B, C_out//4, H, W)
    拼接: 4 * (C_out//4) = C_out，得到 (B, C_out, H, W)

    mid = max(C_in // reduction, 8)
    例: (8, 64, 6, 2250) -> (8, 128, 6, 2250)
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super(MultiScaleConvLayer, self).__init__()
        
        assert out_channels % 4 == 0, "out_channels 必须能被 4 整除"
        branch_channels = out_channels // 4
        mid_channels = max(in_channels // reduction, 8)
        
        # 分支 1: 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, branch_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支 4: AvgPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)