from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Example:
    docs: torch.Tensor  # (M, D)
    query: torch.Tensor  # (D,)
    y: torch.Tensor  # () class label


def make_dataset(
    n: int = 4000,
    *,
    seed: int = 1,
    num_classes: int = 12,
    M: int = 64,
    D: int = 64,
    noise: float = 0.6,
) -> List[Example]:
    """Synthetic task:

    - Each example picks a latent class c.
    - One doc is the relevant memory for c; other docs are distractors.
    - Query is a noisy view of the class prototype.
    - Model must predict c.
    """

    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(num_classes, D, generator=g)

    out: List[Example] = []
    for _ in range(n):
        c = int(torch.randint(0, num_classes, (1,), generator=g).item())

        docs = torch.randn(M, D, generator=g)
        rel_idx = int(torch.randint(0, M, (1,), generator=g).item())
        docs[rel_idx] = proto[c] + noise * torch.randn(D, generator=g)

        # distractors from random other classes
        for i in range(M):
            if i == rel_idx:
                continue
            c2 = int(torch.randint(0, num_classes, (1,), generator=g).item())
            docs[i] = proto[c2] + (noise + 0.2) * torch.randn(D, generator=g)

        query = proto[c] + noise * torch.randn(D, generator=g)
        out.append(Example(docs=docs, query=query, y=torch.tensor(c, dtype=torch.long)))

    return out


def split(data: List[Example], frac: float = 0.85) -> Tuple[List[Example], List[Example]]:
    n = len(data)
    idx = torch.randperm(n)
    k = int(n * frac)
    tr = [data[int(i)] for i in idx[:k]]
    te = [data[int(i)] for i in idx[k:]]
    return tr, te
