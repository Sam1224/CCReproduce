from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


TASKS = ["diagnosis", "severity", "finding"]


@dataclass
class Batch:
    img: torch.Tensor  # (N, D)
    task: torch.Tensor  # (N,)
    y: torch.Tensor  # (N,)


def make_dataset(n: int = 5000, seed: int = 1, d: int = 48) -> Batch:
    g = torch.Generator().manual_seed(seed)

    task = torch.randint(0, len(TASKS), (n,), generator=g)
    img = torch.randn(n, d, generator=g)

    # Each task has its own label function.
    y = torch.zeros(n, dtype=torch.long)

    # diagnosis: 4-way
    diag_w = torch.randn(4, d, generator=g)
    # severity: 3-way
    sev_w = torch.randn(3, d, generator=g)
    # finding: 2-way
    find_w = torch.randn(2, d, generator=g)

    for i in range(n):
        t = int(task[i].item())
        v = img[i]
        if t == 0:
            y[i] = int((diag_w @ v).argmax().item())
        elif t == 1:
            y[i] = int((sev_w @ v).argmax().item())
        else:
            y[i] = int((find_w @ v).argmax().item())

    return Batch(img=img, task=task, y=y)


def split(batch: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = batch.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(img=batch.img[i], task=batch.task[i], y=batch.y[i])

    return sel(idx[:k]), sel(idx[k:])
