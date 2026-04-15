from __future__ import annotations

import torch
from torch import nn


class ConfidenceProbe(nn.Module):
    """Step-wise confidence predictor.

    Input: features of shape [B, T, D]
    Output: logits of shape [B, T]
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = x.shape
        logits = self.net(x.view(bsz * steps, dim)).view(bsz, steps)
        return logits
