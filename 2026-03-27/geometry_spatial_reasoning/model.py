from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_emb = nn.Embedding(2, 8)
        self.net = nn.Sequential(nn.Linear(3 * 2 + 8, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, pts: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pts.flatten(1), self.q_emb(q)], dim=-1)
        return self.net(x)


class GeometryAware(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_emb = nn.Embedding(2, 8)
        self.net = nn.Sequential(nn.Linear(3 * 2 + 3 + 8, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, pts: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Explicit geometry: distances to origin for each point.
        d = (pts ** 2).sum(dim=-1)  # (B,3)
        x = torch.cat([pts.flatten(1), d, self.q_emb(q)], dim=-1)
        return self.net(x)


def acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()
