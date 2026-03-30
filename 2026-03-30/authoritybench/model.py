from __future__ import annotations

import torch
import torch.nn as nn


class AuthorityRanker(nn.Module):
    def __init__(self, d_in: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, K, D)
        return self.net(feats).squeeze(-1)  # (B, K)
