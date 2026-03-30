from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Batch:
    seq: torch.Tensor  # (N, L)
    y: torch.Tensor  # (N,) next item


def make_dataset(n: int = 6000, seed: int = 1, items: int = 200, L: int = 30) -> Batch:
    g = torch.Generator().manual_seed(seed)

    # latent transition matrix (low-rank) for a toy behavior process
    U = torch.randn(items, 16, generator=g)
    V = torch.randn(items, 16, generator=g)
    T = U @ V.t()  # (items, items)
    T = torch.softmax(T, dim=-1)

    seq = torch.zeros(n, L, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        cur = int(torch.randint(0, items, (1,), generator=g).item())
        for t in range(L):
            seq[i, t] = cur
            # sample next
            cur = int(torch.multinomial(T[cur], 1, generator=g).item())
        y[i] = cur

    return Batch(seq=seq, y=y)


def split(b: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = b.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(seq=b.seq[i], y=b.y[i])

    return sel(idx[:k]), sel(idx[k:])
