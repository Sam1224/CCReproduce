from __future__ import annotations

import torch
import torch.nn as nn


class Ranker(nn.Module):
    def __init__(self, in_dim: int = 3 + 3 + 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_dim) -> (B,)
        return self.net(x).squeeze(-1)
