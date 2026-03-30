from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactEncoder(nn.Module):
    def __init__(self, vocab: int = 256, dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., L)
        e = self.emb(x)
        mask = (x != 0).float().unsqueeze(-1)
        return (e * mask).sum(dim=-2) / (mask.sum(dim=-2).clamp_min(1.0))


class MemoryReader(nn.Module):
    def __init__(self, vocab: int = 256, dim: int = 64, num_values: int = 32) -> None:
        super().__init__()
        self.enc = FactEncoder(vocab=vocab, dim=dim)
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, num_values)

    def forward(self, facts: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # facts: (B, K, L), query: (B, L)
        q = self.proj_q(self.enc(query))  # (B, D)
        k = self.proj_k(self.enc(facts))  # (B, K, D)
        att = (k * q.unsqueeze(1)).sum(dim=-1)  # (B, K)
        w = F.softmax(att, dim=-1)
        mem = (w.unsqueeze(-1) * k).sum(dim=1)
        return self.out(mem)
