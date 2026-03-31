from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Episode:
    keys: List[str]
    values: List[str]
    key_emb: torch.Tensor  # (N, D)
    query_key: str
    query_emb: torch.Tensor  # (D,)
    answer: str


def make_dataset(
    n: int = 1200,
    *,
    seed: int = 1,
    N: int = 24,
    D: int = 48,
    vocab: int = 500,
) -> List[Episode]:
    g = torch.Generator().manual_seed(seed)

    # prototype embeddings for keys
    key_proto = torch.randn(vocab, D, generator=g)

    out: List[Episode] = []
    for ep in range(n):
        ids = torch.randperm(vocab, generator=g)[:N].tolist()
        keys = [f"k{idx}" for idx in ids]
        values = [f"v{idx}" for idx in ids]

        emb = key_proto[torch.tensor(ids)] + 0.25 * torch.randn(N, D, generator=g)

        q_i = int(torch.randint(0, N, (1,), generator=g).item())
        query_key = keys[q_i]
        query_emb = emb[q_i] + 0.35 * torch.randn(D, generator=g)

        out.append(
            Episode(
                keys=keys,
                values=values,
                key_emb=emb,
                query_key=query_key,
                query_emb=query_emb,
                answer=values[q_i],
            )
        )

    return out


def split(data: List[Episode], frac: float = 0.85) -> Tuple[List[Episode], List[Episode]]:
    n = len(data)
    idx = torch.randperm(n)
    k = int(n * frac)
    tr = [data[int(i)] for i in idx[:k]]
    te = [data[int(i)] for i in idx[k:]]
    return tr, te
