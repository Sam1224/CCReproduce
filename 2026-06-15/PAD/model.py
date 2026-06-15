from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    n_users: int
    n_items: int
    dim: int = 32


class MatrixFactorization(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.user = nn.Embedding(cfg.n_users, cfg.dim)
        self.item = nn.Embedding(cfg.n_items, cfg.dim)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # returns prob in (0,1)
        x = (self.user(u) * self.item(i)).sum(dim=-1)
        return torch.sigmoid(x)
