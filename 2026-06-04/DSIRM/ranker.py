from __future__ import annotations

import torch
import torch.nn as nn


class SidRanker(nn.Module):
    def __init__(self, num_codebooks: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = 1 + num_codebooks
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1+L)
        return self.mlp(x).squeeze(-1)
