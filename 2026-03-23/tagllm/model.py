from __future__ import annotations

import torch
import torch.nn as nn


class Tagger(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int = 64, hidden: int = 96) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        e = self.emb(x)
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (e * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))
        return self.net(pooled)
