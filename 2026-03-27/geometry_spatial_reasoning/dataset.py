from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


QUESTIONS = ["closest_to_origin", "inside_circle"]


@dataclass
class Batch:
    pts: torch.Tensor  # (N, 3, 2)
    q: torch.Tensor  # (N,)
    y: torch.Tensor  # (N,)


def make_dataset(n: int = 6000, seed: int = 1) -> Batch:
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n, 3, 2, generator=g)
    q = torch.randint(0, len(QUESTIONS), (n,), generator=g)
    y = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        if int(q[i].item()) == 0:
            d = (pts[i] ** 2).sum(dim=-1)
            y[i] = int(torch.argmin(d).item())
        else:
            # inside circle of radius 1.0 for point A
            y[i] = 1 if float((pts[i, 0] ** 2).sum().item()) < 1.0 else 0

    return Batch(pts=pts, q=q.long(), y=y)


def split(b: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = b.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(pts=b.pts[i], q=b.q[i], y=b.y[i])

    return sel(idx[:k]), sel(idx[k:])
