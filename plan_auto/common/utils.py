"""通用工具：随机种子、设备、日志。"""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime
from pathlib import Path


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Tee:
    """同时把 stdout/stderr 写到控制台和日志文件。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._handle = self.path.open("a", encoding="utf-8")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._handle.close()
        return False

    def write(self, text):
        self._stdout.write(text)
        self._handle.write(text)
        self._handle.flush()

    def flush(self):
        self._stdout.flush()
        self._handle.flush()

    def isatty(self):
        return False


class Timer:
    """记录耗时，用于日志。"""

    def __init__(self):
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start
