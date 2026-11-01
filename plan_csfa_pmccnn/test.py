import torch
from models import CSFAPMCCNN

def test():
    # 实例化模型并切换到评估模式
    model = CSFAPMCCNN(in_channels=1, num_classes=2)
    model.eval()
    
    # 模拟测试数据
    x_alpha = torch.randn(16, 1, 11, 4500)
    x_beta = torch.randn(16, 1, 11, 4500)
    
    with torch.no_grad():
        outputs = model(x_alpha, x_beta)
        # 获取预测类别 (模长最大的胶囊)
        predictions = torch.argmax(outputs, dim=1)
        
    print("模型输出概率分布:\n", outputs)
    print("预测类别:", predictions)

if __name__ == '__main__':
    test()