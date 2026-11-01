baseline/
│
├── models/
│   ├── __init__.py
│   ├── ms_conv.py           # 存放 MultiScaleConvLayer 模块
│   ├── attention.py         # 存放 ChannelAttention, SpatialAttention, FrequencyAttention
│   ├── capsule.py           # 存放胶囊层相关代码（后续补充）
│   └── csfa_pmccnn.py       # 存放你现在的这个顶层模型文件 (最推荐)
│
├── utils/
├── train.py
└── test.py