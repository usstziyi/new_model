plan_single_cnn_mlp/

单分支 EEG 语义分类实验（CSFA-PMCCNN 的简化/消融版）

改动：
    1. 单分支：α/β 频段在通道维拼接后只走一条卷积主干（--bands 可指定）
    2. 去掉分支融合后端、LayerNorm、初级胶囊与动态路由
    3. EEG encoder：卷积主干 -> GAP -> 两层 MLP 分类头
    4. EEG-文本对齐：GAP 特征经 768 维线性投影 + L2 归一化，与冻结的 LaBSE
       句向量（每句真实 sentence）对齐
    5. 最终目标：CE_cls + align_weight * InfoNCE，两项都用 CrossEntropyLoss
       （默认 align_weight=0.5、temperature=0.07）

目录结构：
    models/
        __init__.py
        ms_conv.py             # 多尺度卷积（与 CSFA-PMCCNN 相同）
        attention.py           # 通道 / 空间 / 频率注意力
        single_cnn_mlp.py      # 单分支 CNN + MLP + 768 维投影头
        text_encoder.py        # LaBSE 文本编码器（冻结）
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
