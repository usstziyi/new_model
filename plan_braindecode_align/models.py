"""EEG 编码器工厂（对齐实验，完全自包含）。

把 braindecode 的标准模型与自研 PMCNN 统一成同一个接口：

    backbone.features((B, E, T)) -> (B, D)

然后在外面套一层所有模型共享的 ``AlignModel`` 头部：

    (B, D) -> LayerNorm -> Dropout -> Linear(D, proj_hidden) -> GELU
           -> Linear(proj_hidden, embed_dim) -> L2 归一化 -> (B, embed_dim)

这样做的原因是 braindecode 各模型的分类层结构差异很大（``nn.Sequential`` 的
``final_layer``、``Linear``、``ParametrizedLinearWithConstraint``），而它们倒数
第二层的输出维度也各不相同（EEGNet 400、ShallowFBCSPNet 1960、SCCNet 560、
EEGITNet 714、Deep4Net 1000、PMCNN 512）。统一剥掉最后的分类层后，投影头的
输入维度由一次 dummy forward 实测得到，不需要对外暴露模型细节。

与 ``plan_braindecode_baseline/models.py`` 的差异：那里的输出是 ``(B, n_outputs)``
的分类 logits，这里输出的是 ``(B, embed_dim)`` 的 L2 归一化向量，用于与冻结
LaBSE 的句向量做 InfoNCE。

用法::

    from models import available_models, build_align_model

    model = build_align_model("eegnet", n_chans=64, n_times=826, num_classes=10,
                              sfreq=250.0, embed_dim=768)
    z = model(torch.randn(4, 64, 826))   # -> (4, 768)
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import (
    Deep4Net,
    EEGITNet,
    EEGNet,
    SCCNet,
    ShallowFBCSPNet,
)

from pmcnn import PMCNNBackbone

# 名称 -> braindecode 模型类
_BRAINDECODE_MODELS = {
    "eegnet": EEGNet,
    "shallowfbcsp": ShallowFBCSPNet,
    "sccnet": SCCNet,
    "eegitnet": EEGITNet,
    "deep4net": Deep4Net,
}

PMCNN_NAME = "pmcnn"


def available_models() -> list[str]:
    """返回可选的模型名（PMCNN 在前，与分类基线实验的顺序一致）。"""
    return [PMCNN_NAME, *_BRAINDECODE_MODELS]


class BraindecodeBackbone(nn.Module):
    """把 braindecode 模型最后的分类层 ``final_layer`` 换成恒等映射。

    不能按子模块顺序重建 ``nn.Sequential``：``SCCNet`` 的 ``forward`` 是手写的，
    子模块注册顺序与真实执行顺序不同（``activation`` 注册在 ``spatial_conv``
    前面，但实际是在平方与平滑之后才用），重建会静默产生 NaN。

    这里保留原模型对象、只替换 ``final_layer`` 属性，因此对 ``nn.Sequential``
    子类（其余四个模型）和手写 ``forward`` 的模型都成立。特征维度用一次 dummy
    forward 在 ``final_layer`` 的输入上挂 hook 实测，不需要逐模型写适配。
    """

    def __init__(self, model: nn.Module, n_chans: int, n_times: int):
        super().__init__()

        if not hasattr(model, "final_layer"):
            raise ValueError(
                f"{type(model).__name__} 没有 final_layer 属性，无法剥离分类头"
            )

        self.feature_dim = self._probe_dim(model, n_chans, n_times)
        model.final_layer = nn.Identity()
        self.model = model

    @staticmethod
    def _probe_dim(model: nn.Module, n_chans: int, n_times: int) -> int:
        """在 ``final_layer`` 的输入上取一次特征形状（不影响 BN 的 running stats）。"""
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, args, _output):
            captured["feat"] = args[0]

        handle = model.final_layer.register_forward_hook(hook)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(torch.zeros(2, n_chans, n_times))
        finally:
            handle.remove()
            model.train(was_training)

        if "feat" not in captured:
            raise RuntimeError(
                f"{type(model).__name__} 的 final_layer 在 forward 中未被调用"
            )
        return int(captured["feat"][0].numel())

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.model(x), 1)


class AlignModel(nn.Module):
    """主干 + 共享对齐头部。

    输入:  ``(B, E, T)``      原始 EEG
    输出:  ``(B, embed_dim)`` 已 L2 归一化

    额外接口（训练时复用，避免重复前向）：
        ``features(x)``    -> ``(B, D)`` 主干池化特征
        ``embed(feats)``   -> ``(B, embed_dim)`` 归一化向量
        ``classify(feats)``-> ``(B, num_classes)`` 仅辅助 CE 使用
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        embed_dim: int = 768,
        proj_hidden: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.feature_dim = feature_dim

        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, embed_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_heads()

    def _init_heads(self) -> None:
        """头部单独初始化：主干的 std=0.01 对对齐任务偏小，投影头用 xavier。"""
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        z = self.projection(self.dropout(self.norm(features)))
        return F.normalize(z, p=2, dim=-1)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.features(x))


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """只保留目标构造函数真正接受的参数，避免不同模型签名差异导致报错。"""
    params = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def build_align_model(
    name: str,
    n_chans: int,
    n_times: int,
    num_classes: int,
    sfreq: float,
    embed_dim: int = 768,
    proj_hidden: int = 512,
    drop_prob: float = 0.5,
    use_electrode_attn: bool = False,
    e_out: int = 16,
    c0: int = 64,
    stem_k: int = 7,
) -> AlignModel:
    """构造 ``(B, E, T) -> (B, embed_dim)`` 的对齐模型。

    Args:
        name: ``available_models()`` 中的名字。
        n_chans: 电极数（E）。
        n_times: 每个试次的采样点数（T）。
        num_classes: 类别数，仅用于辅助分类头与构造 braindecode 模型。
        sfreq: 采样率。
        embed_dim: 输出嵌入维度，应与文本编码器一致（LaBSE 为 768）。
        proj_hidden: 投影头隐层维度。
        drop_prob: 模型内部 dropout 概率（不支持该参数的模型自动忽略）。
        use_electrode_attn: 仅对 PMCNN 生效，是否在虚拟电极轴上做注意力池化。
        e_out: 仅对 PMCNN 生效，Stem 压缩后的虚拟电极数。
        c0: 仅对 PMCNN 生效，Stem 时间升维后的通道数。
        stem_k: 仅对 PMCNN 生效，Stem 时间卷积核长。
    """
    name = name.lower()

    if name == PMCNN_NAME:
        backbone = PMCNNBackbone(
            E_in=n_chans,
            E_out=e_out,
            C0=c0,
            stem_k=stem_k,
            use_electrode_attn=use_electrode_attn,
        )
    elif name in _BRAINDECODE_MODELS:
        cls = _BRAINDECODE_MODELS[name]
        kwargs = _filter_kwargs(
            cls,
            dict(
                n_chans=n_chans,
                n_outputs=num_classes,
                n_times=n_times,
                sfreq=sfreq,
                drop_prob=drop_prob,
            ),
        )
        backbone = BraindecodeBackbone(cls(**kwargs), n_chans, n_times)
    else:
        raise ValueError(f"未知模型 {name!r}，可选: {available_models()}")

    return AlignModel(
        backbone=backbone,
        feature_dim=backbone.feature_dim,
        num_classes=num_classes,
        embed_dim=embed_dim,
        proj_hidden=proj_hidden,
    )


if __name__ == "__main__":
    """快速自测：逐个构造并跑一次前向，打印特征维度、输出形状与参数量。"""
    B, E, T, C, D = 4, 64, 826, 10, 768
    x = torch.randn(B, E, T)
    for name in available_models():
        model = build_align_model(
            name, n_chans=E, n_times=T, num_classes=C, sfreq=250.0, embed_dim=D
        )
        params = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            feats = model.features(x)
            z = model(x)
        print(
            f"{name:15s} feature={model.feature_dim:5d} "
            f"feats={tuple(feats.shape)} embed={tuple(z.shape)} "
            f"|z|={z.norm(dim=-1)[0]:.4f} params={params:,}"
        )
