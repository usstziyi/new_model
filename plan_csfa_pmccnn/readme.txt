plan_csfa_pmccnn/
│
├── models/
│   ├── __init__.py
│   ├── ms_conv.py           # 存放 MultiScaleConvLayer 模块
│   ├── attention.py         # 存放 ChannelAttention, SpatialAttention, FrequencyAttention
│   ├── capsule.py           # 存放胶囊层相关代码（后续补充）
│   ├── text_encoder.py      # 存放 LaBSE 文本编码器
│   └── csfa_pmccnn.py       # 存放你现在的这个顶层模型文件 (最推荐)
│
├── band.py                  # α/β 带通滤波与首尾裁剪
├── dataset.py               # Chisco EEG Dataset（α/β 频段，供模型双分支）
├── train.py
└── test.py

实验流程（在项目根目录运行）：
    # reading / imagine 试次长度不同，需分别训练
    uv run plan_csfa_pmccnn/train.py --task reading
    uv run plan_csfa_pmccnn/train.py --task imagine

    # 用留出 run 评估已保存的 checkpoint
    uv run plan_csfa_pmccnn/test.py --checkpoint outputs/csfa_pmccnn/<run>/best.pt

数据划分：day 1-3（run 1-27）训练，day 4（28-36）验证，day 5（37-45）测试。
