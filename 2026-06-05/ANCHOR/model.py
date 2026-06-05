from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFRecommender(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user(user_ids)
        v = self.item(item_ids)
        return (u * v).sum(dim=-1)


class NoiseRecognizer(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2),
        )
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user(user_ids), self.item(item_ids)], dim=-1)
        return self.mlp(x)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()
