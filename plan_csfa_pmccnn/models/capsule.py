import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import einsum  # 强烈推荐用 einsum，语义极其清晰


class PrimaryCaps(nn.Module):
    """初级胶囊层"""
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsule_dim = capsule_dim
        self.out_capsules = out_capsules
        
        # 用卷积生成胶囊特征
        # 输出通道数为 out_capsules * capsule_dim
        self.conv = nn.Conv2d(in_channels, out_capsules * capsule_dim, 
                              kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        out = self.conv(x) 
        # 此时 out 形状: (B, out_capsules * capsule_dim, H', W')
        
        # 【核心修改】使用 einops 一行完成拆分与重排
        # 逻辑：把通道维 (n d) 拆开，把空间维 (h w) 合并，最后把胶囊维度 d 放到最后
        out = rearrange(out, 'b (n d) h w -> b (n h w) d', n=self.out_capsules, d=self.capsule_dim)
        # 此时 out 形状: (B, N, capsule_dim)，其中 N = out_capsules * H' * W'
        
        # Squash 非线性
        return self.squash(out)

    @staticmethod
    def squash(s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        norm_squared = norm ** 2
        return (norm_squared / (1 + norm_squared)) * (s / (norm + 1e-8))




class DigitCaps(nn.Module):
    """分类胶囊层 (带动态路由)"""
    def __init__(self, in_caps, in_dim, num_caps, out_dim, num_routing=3):
        super(DigitCaps, self).__init__()
        self.in_caps = in_caps # 输入胶囊的数量
        self.in_dim = in_dim # 输入胶囊的维度
        self.num_caps = num_caps # 输出胶囊的数量
        self.out_dim = out_dim # 输出胶囊的维度
        self.num_routing = num_routing # 路由迭代次数
        
        # 变换矩阵 W: (1, in_caps, num_caps, out_dim, in_dim)
        # 使用标准正态分布初始化，并乘以一个缩小系数（这也是原版的做法）
        self.W = nn.Parameter(torch.randn(1, in_caps, num_caps, out_dim, in_dim) * 0.01)

    def forward(self, x):
        # x: (B, in_caps, in_dim)
        B = x.size(0)
        
        # 1. 计算预测向量 u_hat: (B, in_caps, num_caps, out_dim)
        # 解释：对每个 batch(b)，输入胶囊(i)的向量(d)，乘上变换矩阵(i, j, o, d)，输出 (b, i, j, o)
        # 注意：这里 einops 的 einsum 自动处理了 W 的 batch 广播
        # u_hat[b,i,j] 就是"第 i 个低层胶囊针对第 j 个高层胶囊的预测向量"
        # u_hat 形状为 (B, in_caps, num_caps, out_dim)
        u_hat = einsum(self.W, x, 'b i j o d, b i d -> b i j o')
        
        # 2. 动态路由
        b_ij = torch.zeros(B, self.in_caps, self.num_caps, 1).to(x.device)
        
        for i in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2) # 耦合系数 (B, in_caps, num_caps, 1)
            
            # 计算高层胶囊输入 s_j
            # 利用广播机制：(B, in_caps, num_caps, 1) * (B, in_caps, num_caps, out_dim)
            # 高层胶囊 j 收到的"总输入"
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # (B, 1, num_caps, out_dim)
            
            # Squash 得到 v_j
            v_j = self.squash(s_j) # (B, 1, num_caps, out_dim)
            
            if i < self.num_routing - 1:
                # 更新 b_ij (点积一致性)
                # 【修正】：去掉了 detach，让梯度流动更自然
                # 衡量的是"预测向量与当前输出的吻合程度",预测与结果越吻合，下一轮投票权重就越大
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)
                
        return v_j.squeeze(1) # (B, num_caps, out_dim)

    @staticmethod
    def squash(s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        norm_squared = norm ** 2
        return (norm_squared / (1 + norm_squared)) * (s / (norm + 1e-8))