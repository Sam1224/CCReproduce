from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class GALA(nn.Module):
    def __init__(self, feature_dim: int = 12, hidden_dim: int = 32, num_items: int = 48):
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.mm_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.id_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.history_embedding = nn.Embedding(num_items, hidden_dim)
        self.history_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.behavior_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_items))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(query), dim=-1)

    def encode_mm(self, mm_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mm_encoder(mm_features), dim=-1)

    def encode_id(self, id_features: torch.Tensor) -> torch.Tensor:
        return self.id_encoder(id_features)

    def stage1_triplet_loss(self, query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
        query_repr = self.encode_query(query)
        positive_repr = self.encode_mm(positive)
        negative_repr = self.encode_mm(negative)
        positive_score = (query_repr * positive_repr).sum(dim=-1)
        negative_score = (query_repr * negative_repr).sum(dim=-1)
        return F.relu(margin - positive_score + negative_score).mean()

    def history_context(self, history: torch.Tensor) -> torch.Tensor:
        embedded = self.history_embedding(history)
        _, hidden = self.history_gru(embedded)
        return hidden.squeeze(0)

    def stage2_alignment_loss(self, history: torch.Tensor, query: torch.Tensor, target: torch.Tensor, reward: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        history_repr = self.history_context(history)
        query_repr = self.encode_query(query)
        logits = self.behavior_head(torch.cat([history_repr, query_repr], dim=-1))
        token_loss = F.cross_entropy(logits, target, reduction="none")
        weights = 1.0 + reward
        loss = (weights * token_loss).mean()
        prediction = logits.argmax(dim=-1)
        accuracy = (prediction == target).float().mean().item()
        return loss, {"stage2_acc": accuracy}

    def rank_logits(
        self,
        history: torch.Tensor,
        query: torch.Tensor,
        catalog_mm: torch.Tensor,
        catalog_id: torch.Tensor,
    ) -> torch.Tensor:
        history_repr = self.history_context(history)
        query_repr = self.encode_query(query)
        context = history_repr + query_repr
        mm_repr = self.encode_mm(catalog_mm).unsqueeze(0).expand(query.size(0), -1, -1)
        id_repr = self.encode_id(catalog_id).unsqueeze(0).expand(query.size(0), -1, -1)
        context_expand = context.unsqueeze(1).expand_as(mm_repr)
        gate = self.gate(torch.cat([context_expand, mm_repr], dim=-1))
        fused = gate * mm_repr + (1.0 - gate) * id_repr
        return torch.einsum("bd,bnd->bn", context, fused)

    def stage3_ranking_loss(
        self,
        history: torch.Tensor,
        query: torch.Tensor,
        target: torch.Tensor,
        catalog_mm: torch.Tensor,
        catalog_id: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self.rank_logits(history, query, catalog_mm, catalog_id)
        ce = F.cross_entropy(logits, target, reduction="none")
        loss = ((1.0 + 0.5 * reward) * ce).mean()
        top1 = logits.argmax(dim=-1)
        recall = (top1 == target).float().mean().item()
        return loss, {"recall@1": recall}

    @torch.no_grad()
    def evaluate(self, history: torch.Tensor, query: torch.Tensor, target: torch.Tensor, catalog_mm: torch.Tensor, catalog_id: torch.Tensor) -> Dict[str, float]:
        logits = self.rank_logits(history, query, catalog_mm, catalog_id)
        top1 = logits.argmax(dim=-1)
        top5 = logits.topk(k=5, dim=-1).indices
        recall1 = (top1 == target).float().mean().item()
        recall5 = (top5 == target.unsqueeze(1)).any(dim=1).float().mean().item()
        return {"recall@1": recall1, "recall@5": recall5}
