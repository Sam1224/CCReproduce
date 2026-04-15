from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class Batch:
    input_ids: torch.LongTensor  # [B, T]
    loss_mask: torch.BoolTensor  # [B, T]
    prompt_len: int


def pad_2d(seqs: list[list[int]], pad_id: int) -> torch.LongTensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out
