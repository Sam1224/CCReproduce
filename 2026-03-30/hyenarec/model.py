from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaBlock(nn.Module):
    def __init__(self, d: int, k: int = 9) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d)
        self.pw = nn.Conv1d(d, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y = x.transpose(1, 2)
        y = self.pw(torch.tanh(self.dw(y))).transpose(1, 2)
        return self.norm(x + y)


class HyenaSeqRec(nn.Module):
    def __init__(self, items: int = 200, d: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(items, d)
        self.blocks = nn.ModuleList([HyenaBlock(d=d, k=9) for _ in range(layers)])
        self.head = nn.Linear(d, items)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.emb(seq)  # (B,L,D)
        for b in self.blocks:
            x = b(x)
        h = x[:, -1]  # last token
        return self.head(h)
