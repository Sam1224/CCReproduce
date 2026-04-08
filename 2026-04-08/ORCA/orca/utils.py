from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class EvalStats:
    risk: float
    savings: float
    avg_stop_step: float
