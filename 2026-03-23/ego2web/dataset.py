from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class EgoBatch:
    seq: torch.Tensor  # (N, L)
    y: torch.Tensor  # (N,)


def make_dataset(n: int = 3000, seed: int = 1, vocab: int = 64, num_tasks: int = 8, L: int = 24) -> EgoBatch:
    g = torch.Generator().manual_seed(seed)

    # Each task has a signature pattern token (anchor) that appears in the sequence.
    anchors = torch.randint(2, vocab, (num_tasks,), generator=g)

    y = torch.randint(0, num_tasks, (n,), generator=g)
    seq = torch.randint(2, vocab, (n, L), generator=g)

    for i in range(n):
        # Insert anchor token at a random position.
        pos = int(torch.randint(0, L, (1,), generator=g).item())
        seq[i, pos] = anchors[int(y[i].item())]

    return EgoBatch(seq=seq, y=y)


def split(batch: EgoBatch, frac: float = 0.85) -> Tuple[EgoBatch, EgoBatch]:
    n = batch.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> EgoBatch:
        return EgoBatch(seq=batch.seq[i], y=batch.y[i])

    return sel(idx[:k]), sel(idx[k:])
