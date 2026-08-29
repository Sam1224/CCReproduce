from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128, embedding_dim: int = 96) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).float().unsqueeze(-1)
        embedded_tokens = self.embedding(token_ids)
        pooled_tokens = (embedded_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled_tokens)


class MultimodalEncoder(nn.Module):
    def __init__(self, vocab_size: int, image_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, hidden_dim)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        text_state = self.text_encoder(token_ids)
        image_state = self.image_encoder(image_features)
        fused_state = self.fusion(torch.cat([text_state, image_state], dim=-1))
        return functional.normalize(fused_state, dim=-1)


class CrossEncoderMatcher(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query_state: torch.Tensor, candidate_state: torch.Tensor, ambiguity: torch.Tensor) -> torch.Tensor:
        cosine_score = functional.cosine_similarity(query_state, candidate_state).unsqueeze(-1)
        ambiguity_feature = ambiguity.unsqueeze(-1)
        pair_features = torch.cat(
            [query_state, candidate_state, query_state * candidate_state, torch.abs(query_state - candidate_state), cosine_score, ambiguity_feature],
            dim=-1,
        )
        return self.scorer(pair_features).squeeze(-1)


@dataclass
class CascadeOutput:
    retrieve_score: torch.Tensor
    match_logit: torch.Tensor
    match_probability: torch.Tensor
    escalate_mask: torch.Tensor
    final_probability: torch.Tensor


class RetrieveMatchEscalate(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_dim: int = 8,
        hidden_dim: int = 128,
        lower_threshold: float = 0.35,
        upper_threshold: float = 0.65,
    ) -> None:
        super().__init__()
        self.encoder = MultimodalEncoder(vocab_size, image_dim, hidden_dim)
        self.matcher = CrossEncoderMatcher(hidden_dim)
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def encode_pair(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        query_state = self.encoder(batch["query_ids"], batch["query_image"])
        candidate_state = self.encoder(batch["candidate_ids"], batch["candidate_image"])
        return query_state, candidate_state

    def forward(self, batch: Dict[str, torch.Tensor]) -> CascadeOutput:
        query_state, candidate_state = self.encode_pair(batch)
        retrieve_score = functional.cosine_similarity(query_state, candidate_state)
        match_logit = self.matcher(query_state, candidate_state, batch["ambiguity"])
        match_probability = torch.sigmoid(match_logit)
        escalate_mask = (match_probability > self.lower_threshold) & (match_probability < self.upper_threshold)
        final_probability = self.agentic_vlm_resolution(match_probability, retrieve_score, batch["ambiguity"], escalate_mask)
        return CascadeOutput(retrieve_score, match_logit, match_probability, escalate_mask, final_probability)

    def agentic_vlm_resolution(
        self,
        match_probability: torch.Tensor,
        retrieve_score: torch.Tensor,
        ambiguity: torch.Tensor,
        escalate_mask: torch.Tensor,
    ) -> torch.Tensor:
        resolved_probability = match_probability.clone()
        heuristic_vote = torch.sigmoid(2.2 * retrieve_score + 0.8 * (1.0 - ambiguity))
        resolved_probability[escalate_mask] = 0.5 * match_probability[escalate_mask] + 0.5 * heuristic_vote[escalate_mask]
        return resolved_probability

    def loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        output = self.forward(batch)
        labels = batch["label"]
        retrieve_loss = functional.binary_cross_entropy_with_logits(output.retrieve_score * 6.0, labels)
        teacher_probability = torch.where(labels > 0.5, torch.full_like(labels, 0.92), torch.full_like(labels, 0.08))
        match_loss = functional.binary_cross_entropy_with_logits(output.match_logit, teacher_probability)
        total_loss = retrieve_loss + match_loss
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "retrieve_loss": float(retrieve_loss.detach().cpu()),
            "match_loss": float(match_loss.detach().cpu()),
            "escalation_rate": float(output.escalate_mask.float().mean().detach().cpu()),
        }
        return total_loss, metrics
