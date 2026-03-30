from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Example:
    q: torch.Tensor  # (D,)
    docs: torch.Tensor  # (K, D)
    y: torch.Tensor  # (K,) 0/1 relevance


def make_dataset(n: int = 2000, seed: int = 1, K: int = 8, D: int = 64) -> List[Example]:
    g = torch.Generator().manual_seed(seed)

    # latent topic vectors
    topics = torch.randn(50, D, generator=g)

    out: List[Example] = []
    for _ in range(n):
        t = int(torch.randint(0, topics.shape[0], (1,), generator=g).item())
        q = topics[t] + 0.4 * torch.randn(D, generator=g)

        docs = torch.randn(K, D, generator=g)
        y = torch.zeros(K, dtype=torch.long)

        # choose 1-2 relevant docs
        rel_count = int(torch.randint(1, 3, (1,), generator=g).item())
        rel_idx = torch.randperm(K, generator=g)[:rel_count]

        for i in range(K):
            if int(i in rel_idx):
                docs[i] = topics[t] + 0.5 * torch.randn(D, generator=g)
                y[i] = 1
            else:
                # distractor from other topic
                t2 = int(torch.randint(0, topics.shape[0], (1,), generator=g).item())
                docs[i] = topics[t2] + 0.7 * torch.randn(D, generator=g)

        out.append(Example(q=q, docs=docs, y=y))

    return out


def split(data: List[Example], frac: float = 0.85) -> Tuple[List[Example], List[Example]]:
    n = len(data)
    idx = torch.randperm(n)
    k = int(n * frac)
    tr = [data[int(i)] for i in idx[:k]]
    te = [data[int(i)] for i in idx[k:]]
    return tr, te
