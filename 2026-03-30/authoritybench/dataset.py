from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Example:
    feats: torch.Tensor  # (K, 3) [relevance, authority, noise]
    label: torch.Tensor  # (K,) 1 if trusted


def make_dataset(n: int = 3000, seed: int = 1, K: int = 6) -> List[Example]:
    g = torch.Generator().manual_seed(seed)
    out: List[Example] = []

    for _ in range(n):
        rel = torch.rand(K, generator=g)
        auth = torch.rand(K, generator=g)
        noise = 0.05 * torch.randn(K, generator=g)

        # ground truth: prefer items that are both relevant and authoritative
        score = rel + 0.8 * auth
        best = int(torch.argmax(score).item())

        label = torch.zeros(K, dtype=torch.float32)
        label[best] = 1.0

        feats = torch.stack([rel, auth, noise], dim=-1)
        out.append(Example(feats=feats, label=label))

    return out


def split(data: List[Example], frac: float = 0.85) -> Tuple[List[Example], List[Example]]:
    n = len(data)
    idx = torch.randperm(n)
    k = int(n * frac)
    tr = [data[int(i)] for i in idx[:k]]
    te = [data[int(i)] for i in idx[k:]]
    return tr, te
