from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Batch:
    seq: torch.Tensor  # (N, L)
    y: torch.Tensor  # (N,)


def make_dataset(n: int = 6000, seed: int = 1, vocab: int = 20, L: int = 16) -> Batch:
    g = torch.Generator().manual_seed(seed)

    seq = torch.randint(0, vocab, (n, L), generator=g)

    # Label: center token, but disambiguate with context parity (toy fine-grain).
    center = L // 2
    ctx = seq[:, :center].sum(dim=1)
    y = (seq[:, center] + (ctx % 2) * vocab).long()  # 2*vocab classes

    return Batch(seq=seq.long(), y=y)


def split(b: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = b.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(seq=b.seq[i], y=b.y[i])

    return sel(idx[:k]), sel(idx[k:])
