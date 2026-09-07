from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TradeUpOutput:
    logits: torch.Tensor
    tradeup_score: torch.Tensor
    pair_embedding: torch.Tensor
    predicted_rationale: torch.Tensor


class PairFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256, num_product_types: int = 12) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(num_product_types, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, base: torch.Tensor, candidate: torch.Tensor, product_type: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([base, candidate, candidate - base, base * candidate], dim=-1)
        return self.proj(pair) + self.type_embedding(product_type)


class ProductTypeAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 256, bottleneck: int = 32) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(torch.relu(self.down(hidden)))


class TradeUpStudent(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256, rationale_dim: int = 768, num_product_types: int = 12) -> None:
        super().__init__()
        self.encoder = PairFeatureEncoder(embedding_dim, hidden_dim, num_product_types)
        self.classifier = nn.Linear(hidden_dim, 4)
        self.rationale_head = nn.Linear(hidden_dim, rationale_dim)
        self.adapter = ProductTypeAdapter(hidden_dim)
        self.use_adapter = False

    def freeze_backbone_for_pt_ttt(self) -> None:
        for module in [self.encoder, self.classifier, self.rationale_head]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.use_adapter = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> TradeUpOutput:
        pair_embedding = self.encoder(batch["base_embedding"], batch["candidate_embedding"], batch["product_type"])
        if self.use_adapter:
            pair_embedding = self.adapter(pair_embedding)
        logits = self.classifier(pair_embedding)
        tradeup_score = torch.softmax(logits, dim=-1)[:, 3]
        rationale = self.rationale_head(pair_embedding)
        return TradeUpOutput(logits, tradeup_score, pair_embedding, rationale)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    sim = features @ features.T / temperature
    eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = same & ~eye
    sim = sim.masked_fill(eye, -1e4)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = positive_mask.float().sum(dim=1).clamp_min(1.0)
    return -(log_prob * positive_mask.float()).sum(dim=1).div(positives).mean()


def tradeup_losses(output: TradeUpOutput, batch: Dict[str, torch.Tensor], rationale_weight: float = 0.15, contrastive_weight: float = 0.05) -> Tuple[torch.Tensor, Dict[str, float]]:
    classification = F.cross_entropy(output.logits, batch["label"])
    alignment = F.mse_loss(F.normalize(output.predicted_rationale, dim=-1), F.normalize(batch["rationale_embedding"], dim=-1))
    contrastive = supervised_contrastive_loss(output.pair_embedding, batch["label"])
    loss = classification + rationale_weight * alignment + contrastive_weight * contrastive
    return loss, {
        "classification": float(classification.detach()),
        "rationale_alignment": float(alignment.detach()),
        "contrastive": float(contrastive.detach()),
    }
