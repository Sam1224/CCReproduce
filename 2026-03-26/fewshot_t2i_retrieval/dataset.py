from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Pairs:
    img: torch.Tensor  # (N, D)
    txt: torch.Tensor  # (N, D)
    cls: torch.Tensor  # (N,)


def make_pairs(n: int, seed: int, num_classes: int, d: int = 32, noise: float = 0.3) -> Pairs:
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(num_classes, d, generator=g)
    cls = torch.randint(0, num_classes, (n,), generator=g)

    img = proto[cls] + noise * torch.randn(n, d, generator=g)
    txt = proto[cls] + noise * torch.randn(n, d, generator=g)
    return Pairs(img=img, txt=txt, cls=cls)


def split(p: Pairs, frac: float = 0.8) -> Tuple[Pairs, Pairs]:
    n = p.cls.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Pairs:
        return Pairs(img=p.img[i], txt=p.txt[i], cls=p.cls[i])

    return sel(idx[:k]), sel(idx[k:])
