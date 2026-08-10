from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

MergeMode = Literal["mean", "recency", "attention"]


@dataclass
class TM20KConfig:
    num_items: int = 2048
    max_seq_len: int = 512
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    merged_len: int = 128
    merge_mode: MergeMode = "recency"


class TokenMerger(nn.Module):
    def __init__(self, d_model: int, output_len: int, mode: MergeMode = "recency"):
        super().__init__()
        self.output_len = output_len
        self.mode = mode
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = token_embeddings.shape
        if seq_len <= self.output_len:
            return token_embeddings
        padded_len = self.output_len * ((seq_len + self.output_len - 1) // self.output_len)
        pad = padded_len - seq_len
        if pad:
            token_embeddings = F.pad(token_embeddings, (0, 0, 0, pad))
        group = padded_len // self.output_len
        chunks = token_embeddings.view(batch_size, self.output_len, group, dim)
        if self.mode == "mean":
            return chunks.mean(dim=2)
        if self.mode == "recency":
            weights = torch.linspace(0.35, 1.0, group, device=token_embeddings.device)
            weights = weights / weights.sum()
            return (chunks * weights.view(1, 1, group, 1)).sum(dim=2)
        scores = self.scorer(chunks).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return (chunks * weights.unsqueeze(-1)).sum(dim=2)


class FullAttentionRanker(nn.Module):
    def __init__(self, config: TM20KConfig, use_token_merge: bool):
        super().__init__()
        self.config = config
        self.use_token_merge = use_token_merge
        self.item_embedding = nn.Embedding(config.num_items + 1, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_seq_len + 2, config.d_model)
        self.merger = TokenMerger(config.d_model, config.merged_len, config.merge_mode)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, sequence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seq_emb = self.item_embedding(sequence)
        if self.use_token_merge:
            seq_emb = self.merger(seq_emb)
        target_emb = self.item_embedding(target).unsqueeze(1)
        tokens = torch.cat([seq_emb, target_emb], dim=1)
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        tokens = tokens + self.position_embedding(positions)
        encoded = self.norm(self.encoder(tokens))
        target_state = encoded[:, -1]
        pooled = encoded[:, :-1].mean(dim=1)
        maxed = encoded[:, :-1].amax(dim=1)
        return self.head(torch.cat([target_state, pooled, maxed], dim=-1)).squeeze(-1)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.55,
    temperature: float = 2.0,
) -> torch.Tensor:
    hard = F.binary_cross_entropy_with_logits(student_logits, labels)
    teacher_prob = torch.sigmoid(teacher_logits.detach() / temperature)
    soft = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_prob)
    return alpha * hard + (1 - alpha) * soft * temperature * temperature
