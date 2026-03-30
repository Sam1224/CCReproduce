from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Batch:
    facts: torch.Tensor  # (N, K, L)
    query: torch.Tensor  # (N, L)
    answer: torch.Tensor  # (N,) value id


def make_dataset(n: int = 4000, seed: int = 1, K: int = 8, L: int = 12, vocab: int = 256, num_values: int = 32) -> Batch:
    g = torch.Generator().manual_seed(seed)

    # Represent each fact as tokens; its value is an integer class.
    value_tokens = torch.randint(2, vocab, (num_values, L), generator=g)

    facts = torch.zeros(n, K, L, dtype=torch.long)
    query = torch.zeros(n, L, dtype=torch.long)
    answer = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        vals = torch.randint(0, num_values, (K,), generator=g)
        facts[i] = value_tokens[vals]

        tgt = int(torch.randint(0, K, (1,), generator=g).item())
        answer[i] = int(vals[tgt].item())

        # Query is a noisy transform of the target fact.
        q = value_tokens[answer[i]].clone()
        noise = torch.randint(2, vocab, (L // 3,), generator=g)
        q[: len(noise)] = noise
        query[i] = q

    return Batch(facts=facts, query=query, answer=answer)


def split(batch: Batch, frac: float = 0.85) -> Tuple[Batch, Batch]:
    n = batch.answer.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> Batch:
        return Batch(facts=batch.facts[i], query=batch.query[i], answer=batch.answer[i])

    return sel(idx[:k]), sel(idx[k:])
