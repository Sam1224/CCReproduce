from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Batch:
    video: torch.Tensor  # (N, L) digits
    step: torch.Tensor  # (N,) step label
    y: torch.Tensor  # (N,) answer label


def make_dataset(n: int = 5000, seed: int = 1, L: int = 16) -> Batch:
    g = torch.Generator().manual_seed(seed)
    video = torch.randint(0, 10, (n, L), generator=g)

    # Structured step: sum bucket (0..4)
    s = video.sum(dim=1)
    step = torch.clamp(s // 8, 0, 4).long()

    # Final answer: is sum >= threshold(step-dependent)
    thresh = (step + 1) * 12
    y = (s >= thresh).long()

    return Batch(video=video.long(), step=step, y=y)


def split(batch: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = batch.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(video=batch.video[i], step=batch.step[i], y=batch.y[i])

    return sel(idx[:k]), sel(idx[k:])
