import torch
import torch.nn as nn
import torch.optim as optim
from models import CSFAPMCCNN


class MarginLoss(nn.Module):
    """胶囊网络 Margin Loss (论文公式 2.24)

    L_k = T_k * max(0, m+ - ||v_k||)^2
        + λ * (1 - T_k) * max(0, ||v_k|| - m-)^2
    """

    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, v_norm, labels):
        # v_norm: (B, num_classes) 各胶囊的模长；labels: (B,) 真实类别索引
        t = torch.zeros_like(v_norm)
        t.scatter_(1, labels.view(-1, 1), 1.0)

        # 正样本损失
        present_loss = t * torch.clamp(self.m_plus - v_norm, min=0) ** 2
        # 负样本损失
        absent_loss = self.lambda_ * (1 - t) * torch.clamp(v_norm - self.m_minus, min=0) ** 2
        return (present_loss + absent_loss).sum(dim=1).mean()


def train():
    # 固定随机种子，保证每次运行结果一致
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 超参数设定
    batch_size = 8
    epochs = 10
    learning_rate = 1e-3

    # 2. 模拟数据加载 (B, C, E, T) -> (8, 1, 122, 1651)
    # C 是特征通道维（模型 in_channels=1），E 是电极数，T 是时间采样点数
    in_channels = 1
    num_classes = 2
    x_alpha = torch.randn(batch_size, in_channels, 122, 1651)
    x_beta = torch.randn(batch_size, in_channels, 122, 1651)
    
    # 3. 实例化模型、损失函数、优化器
    model = CSFAPMCCNN(in_channels, num_classes)
    
    # 胶囊层输出的是各类别的模长，使用 Margin Loss (论文公式 2.24)
    criterion = MarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 标签 0, 1, ..., num_classes-1
    labels = torch.randint(0, num_classes, (batch_size,)) 
    
    # 4. 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(x_alpha, x_beta) # 输出形状 (B, num_classes)
        
        # 胶囊模长直接作为预测概率，交给 Margin Loss 计算
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()