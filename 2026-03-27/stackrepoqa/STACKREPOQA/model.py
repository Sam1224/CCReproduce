from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_]+", text.lower()) if t]


def hash_bow(tokens: List[str], dim: int) -> torch.Tensor:
    # Simple hashing trick: bag-of-words into a fixed vector.
    out = torch.zeros(dim)
    for t in tokens:
        h = hash(t) % dim
        out[h] += 1.0
    return out


class TextEncoder(nn.Module):
    def __init__(self, in_dim: int = 4096, out_dim: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


@dataclass
class BiEncoderBatch:
    q: torch.Tensor  # (B,D)
    d_pos: torch.Tensor  # (B,D)


def make_batch(questions: List[str], docs: List[str], in_dim: int) -> BiEncoderBatch:
    q = torch.stack([hash_bow(tokenize(s), in_dim) for s in questions], dim=0)
    d = torch.stack([hash_bow(tokenize(s), in_dim) for s in docs], dim=0)
    return BiEncoderBatch(q=q, d_pos=d)


def contrastive_loss(q_emb: torch.Tensor, d_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    # In-batch negatives.
    logits = q_emb @ d_emb.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
