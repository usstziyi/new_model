"""Braindecode 标准模型工厂（完全自包含）。

数据管线统一返回 ``(B, E, T)``；braindecode 的模型原生接受这个形状，PMCNN
（见 :mod:`pmcnn`）在 ``forward`` 内部自行补回单通道维，因此工厂不再需要任何
输入形状适配，所有候选模型都是 ``(B, E, T) -> (B, n_outputs)``。

对照用的 PMCNN 源自 ``plan_top10_random/model.py``，结构与初始化一致，只删掉了
Stem 末尾那个秩恒为 1 的 ``feature_expand``（见 :mod:`pmcnn`），参数量
630,570 -> 611,290。

模型清单只挑了 CPU 上跑得动的轻量结构（实测 64 通道 × 1651 点单样本前向
2–12 ms），Deep4Net / EEGConformer / ATCNet / BIOT / BENDR 这类要么太重、要么
依赖标准电极名，暂不纳入。

用法::

    from models import available_models, build_model

    model = build_model("eegnet", n_chans=64, n_times=1651, n_outputs=10, sfreq=500)
    y = model(torch.randn(4, 64, 1651))   # -> (4, 10)
"""

from __future__ import annotations

import inspect

from braindecode.models import (
    Deep4Net,
    EEGITNet,
    EEGNet,
    SCCNet,
    ShallowFBCSPNet,
)

from pmcnn import PMCNN


# 名称 -> 模型类
_BRAINDECODE_MODELS = {
    "eegnet": EEGNet,
    "shallowfbcsp": ShallowFBCSPNet,
    "sccnet": SCCNet,
    "eegitnet": EEGITNet,
    "deep4net": Deep4Net,
}

PMCNN_NAME = "pmcnn"


def available_models() -> list[str]:
    """返回可选的模型名（PMCNN 在前，便于与既有结果对照）。"""
    return [PMCNN_NAME, *_BRAINDECODE_MODELS]


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """只保留目标构造函数真正接受的参数，避免不同模型签名差异导致报错。"""
    params = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def build_model(
    name: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: float,
    drop_prob: float = 0.5,
):
    """构造模型，返回的模块统一接受 ``(B, E, T)`` 输入。

    Args:
        name: ``available_models()`` 中的名字。
        n_chans: 电极数（Stem 的 E_in）。
        n_times: 每个试次的采样点数。
        n_outputs: 类别数。
        sfreq: 采样率。
        drop_prob: 模型内部 dropout 概率（不支持该参数的模型自动忽略）。
    """
    name = name.lower()
    if name == PMCNN_NAME:
        return PMCNN(num_classes=n_outputs, E_in=n_chans, E_out=16, C0=64, stem_k=7)

    if name not in _BRAINDECODE_MODELS:
        raise ValueError(f"未知模型 {name!r}，可选: {available_models()}")

    cls = _BRAINDECODE_MODELS[name]
    kwargs = _filter_kwargs(
        cls,
        dict(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            drop_prob=drop_prob,
        ),
    )
    return cls(**kwargs)


if __name__ == "__main__":
    """快速自测：逐个构造并跑一次前向，打印输出形状与参数量。"""
    import torch

    B, E, T, C = 4, 64, 1651, 10
    x = torch.randn(B, E, T)
    for name in available_models():
        model = build_model(name, n_chans=E, n_times=T, n_outputs=C, sfreq=500.0)
        model.eval()
        with torch.no_grad():
            y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s} out={tuple(y.shape)} params={params:,}")
