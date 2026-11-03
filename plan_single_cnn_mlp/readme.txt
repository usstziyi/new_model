plan_single_cnn_mlp/

单分支 EEG 语义分类实验（CSFA-PMCCNN 的简化/消融版）

改动：
    1. 单分支：α/β 频段在通道维拼接后只走一条卷积主干（--bands 可指定）
    2. 去掉分支融合后端、LayerNorm、初级胶囊与动态路由
    3. 分类头：普通 3×3 卷积 -> 全局平均池化 -> 两层 MLP
    4. 损失函数：交叉熵 CrossEntropyLoss

目录结构：
    models/
        __init__.py
        ms_conv.py             # 多尺度卷积（与 CSFA-PMCCNN 相同）
        attention.py           # 通道 / 空间 / 频率注意力
        single_cnn_mlp.py      # 单分支 CNN + MLP 顶层模型
    band.py                    # α/β/γ 带通滤波与首尾裁剪
    dataset.py                 # Chisco EEG Dataset（按 bands 返回 (C, 122, T)）
    train.py
    test.py

实验流程（在项目根目录运行）：
    # 默认 α+β 双通道输入
    uv run plan_single_cnn_mlp/train.py --task reading
    uv run plan_single_cnn_mlp/train.py --task imagine

    # 严格单频段（单通道）输入
    uv run plan_single_cnn_mlp/train.py --task imagine --bands alpha

    # 用训练时留出的验证集（20% run）重新评估已保存的 checkpoint
    uv run plan_single_cnn_mlp/test.py \
        --checkpoint outputs/single_cnn_mlp/<run>/best.pt

数据划分：按 run 随机划分（默认 80% 训练 / 20% 验证，无测试集），
划分由 --seed（默认 42）固定、可复现；同一个 run 的试次只会出现在一侧。
划分结果与 run 列表会随 best.pt 一并保存，test.py 直接复用。
