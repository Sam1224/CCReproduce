from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class RankTower(nn.Module):
    def __init__(self, user_dim: int = 16, item_dim: int = 24, pxtr_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(user_dim + item_dim + pxtr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user: torch.Tensor, item: torch.Tensor, pxtr: torch.Tensor) -> torch.Tensor:
        return self.scorer(torch.cat([user, item, pxtr], dim=-1)).squeeze(-1)


class DrEM(nn.Module):
    def __init__(self, user_dim: int = 16, item_dim: int = 24, pxtr_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.rank_tower = RankTower(user_dim=user_dim, item_dim=item_dim, pxtr_dim=pxtr_dim, hidden_dim=hidden_dim)

    def score_pair(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        left_score = self.rank_tower(batch["user"], batch["left_item"], batch["left_pxtr"])
        right_score = self.rank_tower(batch["user"], batch["right_item"], batch["right_pxtr"])
        return left_score, right_score

    def robust_loss(self, batch: Dict[str, torch.Tensor], consistency_weight: float = 0.4) -> Tuple[torch.Tensor, Dict[str, float]]:
        left_score, right_score = self.score_pair(batch)
        margin = left_score - right_score

        pxtr_margin = batch["left_pxtr"].mean(dim=-1) - batch["right_pxtr"].mean(dim=-1)
        noise_level = (batch["left_noise"].mean(dim=-1) + batch["right_noise"].mean(dim=-1)).clamp_min(1e-3)
        flip_prob = torch.sigmoid(-pxtr_margin / noise_level)
        sample_weight = 1.0 + flip_prob
        ranking_loss = F.binary_cross_entropy_with_logits(margin, batch["label"], weight=sample_weight)

        left_perturbed = torch.clamp(batch["left_pxtr"] + batch["left_noise"] * torch.randn_like(batch["left_pxtr"]), 0.0, 1.0)
        right_perturbed = torch.clamp(batch["right_pxtr"] + batch["right_noise"] * torch.randn_like(batch["right_pxtr"]), 0.0, 1.0)
        left_aug = self.rank_tower(batch["user"], batch["left_item"], left_perturbed)
        right_aug = self.rank_tower(batch["user"], batch["right_item"], right_perturbed)
        consistency = F.mse_loss(torch.sigmoid(margin), torch.sigmoid(left_aug - right_aug))

        loss = ranking_loss + consistency_weight * consistency
        accuracy = ((margin > 0).float() == batch["label"]).float().mean().item()
        metrics = {
            "pair_acc": accuracy,
            "flip_prob": flip_prob.mean().item(),
            "ranking_loss": ranking_loss.item(),
            "consistency": consistency.item(),
        }
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        left_score, right_score = self.score_pair(batch)
        margin = left_score - right_score
        probability = torch.sigmoid(margin)
        label = batch["label"]
        pair_acc = ((probability > 0.5).float() == label).float().mean().item()
        gauc = torch.where(label > 0.5, probability, 1.0 - probability).mean().item()
        return {"pair_acc": pair_acc, "gauc": gauc}
