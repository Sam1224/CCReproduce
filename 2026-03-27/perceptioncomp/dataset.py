from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Batch:
    video: torch.Tensor  # (N, T, F)
    y: torch.Tensor  # (N,)


def make_dataset(n: int = 5000, seed: int = 1, T: int = 12, F: int = 8) -> Batch:
    g = torch.Generator().manual_seed(seed)

    video = torch.randn(n, T, F, generator=g)

    # Insert a symbolic cue: feature[0] spikes for "blue", feature[1] spikes for "red".
    y = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        t_blue = int(torch.randint(0, T, (1,), generator=g).item())
        t_red = int(torch.randint(0, T, (1,), generator=g).item())
        video[i, t_blue, 0] += 5.0
        video[i, t_red, 1] += 5.0
        y[i] = 1 if t_red > t_blue else 0

    return Batch(video=video, y=y)


def split(b: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = b.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(video=b.video[i], y=b.y[i])

    return sel(idx[:k]), sel(idx[k:])
