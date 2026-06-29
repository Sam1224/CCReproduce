from __future__ import annotations

import torch
import torch.nn as nn


class ScoringModel(nn.Module):
    """A tiny multi-task scorer.

    In production ranking stacks, relevance / monetization scores are often produced
    by separate models. For this repro we train a shared encoder with two heads.

    Outputs:
      - rel_logit: unnormalized relevance logit
      - revenue: positive revenue score
    """

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rel_head = nn.Linear(hidden_dim, 1)
        self.rev_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, N, D] -> (rel_logit, revenue_pred) each [B, N]."""
        b, n, d = x.shape
        h = self.backbone(x.view(b * n, d))
        rel = self.rel_head(h).view(b, n)
        rev = self.rev_head(h).view(b, n)
        return rel, rev
