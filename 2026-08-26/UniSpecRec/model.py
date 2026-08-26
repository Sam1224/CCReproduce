from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UniSpecOutput:
    scores: torch.Tensor  # [B, C]
    collab_scores: torch.Tensor  # [B, C]
    semantic_scores: torch.Tensor  # [B, C]
    gate: torch.Tensor  # [B]


class UniSpecRec(nn.Module):
    """Decoupled CF/semantic towers with late fusion.

    Semantic item features are pre-smoothed by `data.spectral_smooth`, so the
    semantic tower receives a low-pass version rather than raw noisy LLM content.
    """

    def __init__(self, num_users: int, num_items: int, d: int, sem_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.user_cf = nn.Embedding(num_users, d)
        self.item_cf = nn.Embedding(num_items, d)
        self.user_sem = nn.Embedding(num_users, d)
        self.sem_proj = nn.Sequential(nn.Linear(sem_dim, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.gate = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.item_bias = nn.Embedding(num_items, 1)

    @torch.no_grad()
    def init_from_world(self, item_cf: torch.Tensor, user_cf: torch.Tensor, user_sem: torch.Tensor) -> None:
        self.item_cf.weight.copy_(item_cf)
        self.user_cf.weight.copy_(user_cf)
        self.user_sem.weight.copy_(user_sem[:, : self.user_sem.embedding_dim])
        self.item_bias.weight.zero_()

    def forward(self, user_id: torch.Tensor, cand_item_ids: torch.Tensor, semantic_item: torch.Tensor) -> UniSpecOutput:
        u_cf = F.normalize(self.user_cf(user_id), dim=-1)  # [B, d]
        i_cf = F.normalize(self.item_cf(cand_item_ids), dim=-1)  # [B, C, d]
        collab = torch.sum(u_cf[:, None, :] * i_cf, dim=-1)

        u_sem = F.normalize(self.user_sem(user_id), dim=-1)
        i_sem = F.normalize(self.sem_proj(semantic_item), dim=-1)
        semantic = torch.sum(u_sem[:, None, :] * i_sem, dim=-1)

        gate = torch.sigmoid(self.gate(torch.cat([u_cf, u_sem], dim=-1)).squeeze(-1))
        bias = self.item_bias(cand_item_ids).squeeze(-1)
        scores = gate[:, None] * collab + (1.0 - gate)[:, None] * semantic + bias
        return UniSpecOutput(scores=scores, collab_scores=collab, semantic_scores=semantic, gate=gate)


def gather_semantic(world, cand_item_ids: torch.Tensor) -> torch.Tensor:
    return world.semantic_smooth[cand_item_ids]


def decoupling_penalty(out: UniSpecOutput) -> torch.Tensor:
    """Keep towers complementary rather than forcing early alignment."""

    c = out.collab_scores - out.collab_scores.mean(dim=1, keepdim=True)
    s = out.semantic_scores - out.semantic_scores.mean(dim=1, keepdim=True)
    corr = (F.normalize(c, dim=1) * F.normalize(s, dim=1)).sum(dim=1)
    return corr.abs().mean()
