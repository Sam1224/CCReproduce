from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CatEmbedding(nn.Module):
    def __init__(self, n_cat: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_cat, dim)

    def forward(self, cat: torch.Tensor) -> torch.Tensor:
        return self.emb(cat)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DotInteraction(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1, keepdim=True)


class ConcatInteraction(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=-1)


class TemperatureCalibrator(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.log_temp.exp().clamp(min=1e-3)
        return logits / temp


def bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)
