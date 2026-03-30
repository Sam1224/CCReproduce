from __future__ import annotations

import torch
import torch.nn as nn


class TemplateSelector(nn.Module):
    def __init__(self, in_dim: int = 4, num_templates: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_templates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
