from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryAdapter(nn.Module):
    def __init__(self, num_categories: int = 16, embedding_dim: int = 128) -> None:
        super().__init__()
        self.source = nn.Embedding(num_categories, embedding_dim)
        self.target = nn.Embedding(num_categories, embedding_dim)
        self.gate = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.Sigmoid())

    def forward(self, query_vec: torch.Tensor, cand_vec: torch.Tensor, query_category: torch.Tensor, candidate_category: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_bias = self.source(query_category)
        target_bias = self.target(candidate_category)
        gate = self.gate(torch.cat([source_bias, target_bias], dim=-1))
        return query_vec + gate * source_bias, cand_vec + gate * target_bias


class TwoTower(nn.Module):
    def __init__(self, feature_dim: int = 64, embedding_dim: int = 128, num_categories: int = 16) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(feature_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.candidate_tower = nn.Sequential(nn.Linear(feature_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.category_adapter = CategoryAdapter(num_categories, embedding_dim)
        self.calibrator = nn.Linear(embedding_dim * 3, 1)

    def encode(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_tower(batch["query_feature"])
        candidate = self.candidate_tower(batch["candidate_feature"])
        query, candidate = self.category_adapter(query, candidate, batch["query_category"], batch["candidate_category"])
        return F.normalize(query, dim=-1), F.normalize(candidate, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        query, candidate = self.encode(batch)
        features = torch.cat([query, candidate, query * candidate], dim=-1)
        return self.calibrator(features).squeeze(-1)


def companion_loss(model: TwoTower, batch: Dict[str, torch.Tensor], category_prior: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = model(batch)
    supervised = F.binary_cross_entropy_with_logits(logits, batch["label"])
    query, candidate = model.encode(batch)
    in_batch_logits = query @ candidate.T / 0.07
    targets = torch.arange(query.size(0), device=query.device)
    contrastive = F.cross_entropy(in_batch_logits, targets)
    loss = supervised + 0.05 * contrastive
    return loss, {"supervised": float(supervised.detach()), "contrastive": float(contrastive.detach())}
