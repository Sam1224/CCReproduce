from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class Sona(nn.Module):
    def __init__(
        self,
        num_items: int = 96,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_length: int = 12,
        teacher_dim: int = 96,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder_head = nn.Linear(hidden_dim, num_items)
        self.student_rank_head = nn.Linear(hidden_dim, hidden_dim)
        self.teacher_rank_head = nn.Sequential(
            nn.Linear(hidden_dim, teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, num_items),
        )

    def encode(self, session: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(session.size(1), device=session.device).unsqueeze(0)
        hidden = self.item_embedding(session) + self.position_embedding(positions)
        hidden = self.encoder(hidden)
        state = hidden[:, -1]
        return hidden, state

    def student_logits(self, state: torch.Tensor) -> torch.Tensor:
        projected_state = self.student_rank_head(state)
        item_vectors = self.item_embedding.weight
        return torch.matmul(projected_state, item_vectors.t())

    def forward(self, session: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden, state = self.encode(session)
        return {
            "decoder_logits": self.decoder_head(hidden[:, -1]),
            "student_logits": self.student_logits(state),
            "teacher_logits": self.teacher_rank_head(state),
        }

    def loss(self, session: torch.Tensor, target: torch.Tensor, temperature: float = 2.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.forward(session)
        decoder_loss = F.cross_entropy(outputs["decoder_logits"], target)
        student_loss = F.cross_entropy(outputs["student_logits"], target)
        teacher_loss = F.cross_entropy(outputs["teacher_logits"], target)
        distill_loss = F.kl_div(
            F.log_softmax(outputs["student_logits"] / temperature, dim=-1),
            F.softmax(outputs["teacher_logits"].detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        total_loss = decoder_loss + student_loss + teacher_loss + 0.7 * distill_loss
        predictions = outputs["student_logits"].argmax(dim=-1)
        metrics = {
            "top1": (predictions == target).float().mean().item(),
            "decoder_loss": decoder_loss.item(),
            "student_loss": student_loss.item(),
            "teacher_loss": teacher_loss.item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def evaluate(self, session: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        logits = self.forward(session)["student_logits"]
        top1 = logits.argmax(dim=-1)
        top5 = logits.topk(k=5, dim=-1).indices
        top10 = logits.topk(k=10, dim=-1).indices
        target_expanded = target.unsqueeze(1)
        rank_positions = (top10 == target_expanded).float().argmax(dim=1) + 1
        ndcg10 = torch.where(
            (top10 == target_expanded).any(dim=1),
            1.0 / torch.log2(rank_positions.float() + 1.0),
            torch.zeros_like(rank_positions, dtype=torch.float32),
        )
        return {
            "recall@1": (top1 == target).float().mean().item(),
            "recall@5": (top5 == target_expanded).any(dim=1).float().mean().item(),
            "recall@10": (top10 == target_expanded).any(dim=1).float().mean().item(),
            "ndcg@10": ndcg10.mean().item(),
        }
