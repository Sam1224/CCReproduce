from __future__ import annotations

import torch
import torch.nn as nn


class TargetModel(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RewardModel(nn.Module):
    def __init__(self, d: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
