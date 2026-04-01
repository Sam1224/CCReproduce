from __future__ import annotations

import torch
import torch.nn as nn


class SimulaToyModel(nn.Module):
    """Small MLP regressor trained on synthetic reasoning tasks."""

    def __init__(self, *, in_dim: int = 5, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
