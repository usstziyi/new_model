import torch
from models import CSFAPMCCNN


def test():
    # 固定随机种子，保证每次运行的模拟数据一致
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 超参数设定
    batch_size = 8
    # 2. 模拟数据加载 (B, C, E, T) -> (8, 1, 122, 1651)
    # C 是特征通道维（模型 in_channels=1），E 是电极数，T 是时间采样点数
    in_channels = 1
    num_classes = 2
    x_alpha = torch.randn(batch_size, in_channels, 122, 1651)
    x_beta = torch.randn(batch_size, in_channels, 122, 1651)

    # 3. 实例化模型并切换到评估模式
    model = CSFAPMCCNN(in_channels, num_classes)
    model.eval()

    # 4. 前向传播（不需要梯度）
    with torch.no_grad():
        outputs = model(x_alpha, x_beta) # 输出形状 (B, num_classes)
        # 获取预测类别 (模长最大的胶囊)
        predictions = torch.argmax(outputs, dim=1)

    print("模型输出概率分布:\n", outputs)
    print("预测类别:", predictions)


if __name__ == '__main__':
    test()
