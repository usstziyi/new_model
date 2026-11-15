"""实验 02 的模型工厂（braindecode 标准 EEG 模型）。"""

from __future__ import annotations

import inspect

from braindecode.models import EEGITNet, EEGNet, SCCNet, ShallowFBCSPNet

_MODELS = {
    "eegnet": EEGNet,
    "shallowfbcsp": ShallowFBCSPNet,
    "eegitnet": EEGITNet,
    "sccnet": SCCNet,
}


def available_models() -> list[str]:
    return list(_MODELS)


def build_model(name: str, n_chans: int, n_times: int, n_outputs: int, sfreq: float, drop_prob: float = 0.5):
    cls = _MODELS[name.lower()]
    params = inspect.signature(cls.__init__).parameters
    kwargs = {
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": n_outputs,
        "sfreq": sfreq,
        "drop_prob": drop_prob,
    }
    return cls(**{key: value for key, value in kwargs.items() if key in params})
