from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


CONCEPTS = ["molecule", "cell", "galaxy", "circuit", "equation"]


@dataclass
class Pair:
    img: torch.Tensor  # (N, D)
    txt: torch.Tensor  # (N, L)
    y: torch.Tensor  # (N,)


def make_pairs(n: int = 3000, seed: int = 1, d: int = 32, vocab: int = 256, L: int = 10) -> Pair:
    g = torch.Generator().manual_seed(seed)
    num = len(CONCEPTS)

    proto = torch.randn(num, d, generator=g)
    y = torch.randint(0, num, (n,), generator=g)

    img = proto[y] + 0.35 * torch.randn(n, d, generator=g)

    # text tokens: concept-id specific token pattern
    base_tokens = torch.randint(2, vocab, (num, L), generator=g)
    txt = base_tokens[y] + torch.randint(0, 3, (n, L), generator=g)
    txt = txt.clamp(0, vocab - 1).long()

    return Pair(img=img, txt=txt, y=y)


def prompt_tokens(seed: int = 7, vocab: int = 256, L: int = 10) -> torch.Tensor:
    # deterministic pseudo prompts per class
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (len(CONCEPTS), L), generator=g)
