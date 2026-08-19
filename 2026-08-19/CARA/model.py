from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CaraOutput:
    scores: torch.Tensor  # [B, C]
    gate: torch.Tensor  # [B]
    aff_score: torch.Tensor  # [B, C]
    rat_score: torch.Tensor  # [B, C]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CaraModel(nn.Module):
    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        num_categories: int,
        d: int,
        hidden: int = 96,
    ) -> None:
        super().__init__()

        self.user_aff = nn.Embedding(num_users, d)
        self.user_rat = nn.Embedding(num_users, d)
        self.item_emb = nn.Embedding(num_items, d)
        self.cat_emb = nn.Embedding(num_categories, 8)

        # Continuous item features: price, quality
        head_in = d + d + 8 + 2
        self.aff_head = MLP(head_in, hidden_dim=hidden, out_dim=1)
        self.rat_head = MLP(head_in, hidden_dim=hidden, out_dim=1)

        self.gate = nn.Sequential(nn.Linear(d * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @torch.no_grad()
    def init_from_world(
        self,
        *,
        item_emb: torch.Tensor,
        user_aff: torch.Tensor,
        user_rat: torch.Tensor,
    ) -> None:
        self.item_emb.weight.copy_(item_emb)
        self.user_aff.weight.copy_(user_aff)
        self.user_rat.weight.copy_(user_rat)

    def forward(
        self,
        *,
        user_id: torch.Tensor,
        cand_item_ids: torch.Tensor,
        item_category: torch.Tensor,
        item_price: torch.Tensor,
        item_quality: torch.Tensor,
        filter_topk: Optional[int] = 24,
    ) -> CaraOutput:
        # user_id: [B]
        # cand_item_ids: [B, C]
        bsz, cand = cand_item_ids.shape

        u_aff = self.user_aff(user_id)  # [B, d]
        u_rat = self.user_rat(user_id)  # [B, d]

        items = self.item_emb(cand_item_ids)  # [B, C, d]
        cats = self.cat_emb(item_category)  # [B, C, 8]

        cont = torch.stack([item_price, item_quality], dim=-1)  # [B, C, 2]

        # Candidate filtering (coarse constraints)
        coarse_user = (u_aff + u_rat) / 2.0
        coarse_score = torch.sum(items * coarse_user.unsqueeze(1), dim=-1)  # [B, C]

        if filter_topk is not None and filter_topk < cand:
            top_idx = torch.topk(coarse_score, k=filter_topk, dim=-1).indices  # [B, K]
            mask = torch.full((bsz, cand), fill_value=False, device=cand_item_ids.device)
            mask.scatter_(1, top_idx, True)
        else:
            mask = torch.ones((bsz, cand), dtype=torch.bool, device=cand_item_ids.device)

        gate = torch.sigmoid(self.gate(torch.cat([u_aff, u_rat], dim=-1)).squeeze(-1))  # [B]

        # Build per-candidate features.
        u_aff_rep = u_aff.unsqueeze(1).expand(-1, cand, -1)
        u_rat_rep = u_rat.unsqueeze(1).expand(-1, cand, -1)

        feat = torch.cat([u_aff_rep + items, u_rat_rep + items, cats, cont], dim=-1)

        aff = self.aff_head(feat).squeeze(-1)
        rat = self.rat_head(feat).squeeze(-1)

        scores = gate.unsqueeze(1) * aff + (1.0 - gate).unsqueeze(1) * rat
        scores = scores.masked_fill(~mask, -1e9)

        return CaraOutput(scores=scores, gate=gate, aff_score=aff, rat_score=rat)


def boundary_weight(p_correct: torch.Tensor) -> torch.Tensor:
    """Toy boundary-aware weighting.

    We emphasize *borderline* samples where the model is neither confident nor
    consistently correct. This is a lightweight proxy for boundary-aware KTO.

    p_correct: predicted probability of the gold item, shape [B]
    """

    # Peaks at 0.5, small near 0 or 1.
    hardness = 1.0 - torch.abs(p_correct - 0.5) * 2.0
    hardness = torch.clamp(hardness, 0.0, 1.0)
    return 0.5 + 1.5 * hardness


def gather_item_side(world, cand_item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Utility helper to gather item-side metadata by candidate ids."""

    cat = world.item_category[cand_item_ids]
    price = world.item_price[cand_item_ids]
    quality = world.item_quality[cand_item_ids]
    return cat, price, quality
