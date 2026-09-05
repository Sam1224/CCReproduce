from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class COREConfig:
    vocab_size: int
    image_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    temperature: float = 0.07


class TextEncoder(nn.Module):
    def __init__(self, cfg: COREConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(64, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        mask = tokens.eq(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=mask)
        keep = (~mask).float().unsqueeze(-1)
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.norm(pooled)


class COREEmbeddingModel(nn.Module):
    def __init__(self, cfg: COREConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.query_encoder = TextEncoder(cfg)
        self.candidate_text_encoder = TextEncoder(cfg)
        self.image_projection = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )

    def forward(self, query_tokens: torch.Tensor, candidate_tokens: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.query_encoder(query_tokens), dim=-1)
        cand_text = self.candidate_text_encoder(candidate_tokens)
        cand_image = self.image_projection(image_features)
        candidate = F.normalize(self.fusion(torch.cat([cand_text, cand_image], dim=-1)), dim=-1)
        return (query * candidate).sum(dim=-1) / self.cfg.temperature


class COREReranker(nn.Module):
    def __init__(self, cfg: COREConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(3, cfg.hidden_dim)
        self.image_projection = nn.Linear(cfg.image_dim, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 1))

    def forward(self, query_tokens: torch.Tensor, candidate_tokens: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        query_type = torch.zeros_like(query_tokens)
        cand_type = torch.ones_like(candidate_tokens)
        image_type = torch.full((query_tokens.size(0), 1), 2, dtype=torch.long, device=query_tokens.device)
        query = self.token_embedding(query_tokens) + self.type_embedding(query_type)
        candidate = self.token_embedding(candidate_tokens) + self.type_embedding(cand_type)
        image = self.image_projection(image_features).unsqueeze(1) + self.type_embedding(image_type)
        x = torch.cat([query, candidate, image], dim=1)
        encoded = self.encoder(x)
        return self.head(encoded[:, -1]).squeeze(-1)


def _grouped_softmax(values: torch.Tensor, group_id: torch.Tensor, temperature: float) -> Dict[int, torch.Tensor]:
    groups: Dict[int, torch.Tensor] = {}
    for group in group_id.unique(sorted=True):
        mask = group_id.eq(group)
        groups[int(group.item())] = F.softmax(values[mask] / temperature, dim=0)
    return groups


def rank_kl_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, group_id: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    losses = []
    teacher_by_group = _grouped_softmax(teacher_scores.detach(), group_id, temperature)
    for group, teacher_prob in teacher_by_group.items():
        mask = group_id.eq(group)
        student_log_prob = F.log_softmax(student_scores[mask] / temperature, dim=0)
        losses.append(F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2))
    return torch.stack(losses).mean()


def contrastive_level_loss(scores: torch.Tensor, level: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
    losses = []
    for group in group_id.unique(sorted=True):
        mask = group_id.eq(group)
        target = level[mask].argmax().unsqueeze(0)
        losses.append(F.cross_entropy(scores[mask].unsqueeze(0), target))
    return torch.stack(losses).mean()
