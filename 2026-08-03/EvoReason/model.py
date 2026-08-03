from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class EvoReason(nn.Module):
    def __init__(self, num_items: int = 40, num_primitives: int = 4, feature_dim: int = 10, hidden_dim: int = 32, steps: int = 3):
        super().__init__()
        self.steps = steps
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.item_projector = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.history_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.primitive_embedding = nn.Embedding(num_primitives, hidden_dim)
        self.reason_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.primitive_head = nn.Linear(hidden_dim, num_primitives)
        self.recommendation_head = nn.Linear(hidden_dim, hidden_dim)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embedding(history)
        _, hidden = self.history_encoder(embedded)
        return hidden.squeeze(0)

    def teacher_latents(self, primitive_labels: torch.Tensor, history_repr: torch.Tensor) -> torch.Tensor:
        latents = []
        state = history_repr
        for step in range(self.steps):
            primitive_repr = self.primitive_embedding(primitive_labels[:, step])
            state = torch.tanh(state + primitive_repr)
            latents.append(state)
        return torch.stack(latents, dim=1)

    def student_rollout(self, history_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = []
        logits = []
        state = history_repr
        reason_input = history_repr
        for _ in range(self.steps):
            state = self.reason_cell(reason_input, state)
            step_logits = self.primitive_head(state)
            predicted = step_logits.argmax(dim=-1)
            reason_input = self.primitive_embedding(predicted)
            latents.append(state)
            logits.append(step_logits)
        return torch.stack(latents, dim=1), torch.stack(logits, dim=1)

    def forward(self, history: torch.Tensor, item_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_repr = self.encode_history(history)
        student_latents, primitive_logits = self.student_rollout(history_repr)
        final_state = self.recommendation_head(student_latents[:, -1])
        candidate_repr = self.item_projector(item_features)
        scores = torch.einsum("bd,nd->bn", final_state, candidate_repr)
        return scores, primitive_logits, history_repr

    def loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        primitive_labels: torch.Tensor,
        item_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        scores, primitive_logits, history_repr = self.forward(history, item_features)
        teacher_latents = self.teacher_latents(primitive_labels, history_repr)
        student_latents, _ = self.student_rollout(history_repr)

        rec_loss = F.cross_entropy(scores, target)
        primitive_loss = F.cross_entropy(primitive_logits.reshape(-1, primitive_logits.size(-1)), primitive_labels.reshape(-1))
        distill_loss = F.mse_loss(student_latents, teacher_latents)
        top1 = scores.argmax(dim=-1)
        primitive_prediction = primitive_logits.argmax(dim=-1)
        metrics = {
            "recall@1": (top1 == target).float().mean().item(),
            "primitive_acc": (primitive_prediction == primitive_labels).float().mean().item(),
        }
        return rec_loss + 0.6 * primitive_loss + 0.4 * distill_loss, metrics

    @torch.no_grad()
    def evaluate(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        primitive_labels: torch.Tensor,
        item_features: torch.Tensor,
    ) -> Dict[str, float]:
        scores, primitive_logits, _ = self.forward(history, item_features)
        top1 = scores.argmax(dim=-1)
        top5 = scores.topk(k=5, dim=-1).indices
        primitive_prediction = primitive_logits.argmax(dim=-1)
        return {
            "recall@1": (top1 == target).float().mean().item(),
            "recall@5": (top5 == target.unsqueeze(1)).any(dim=1).float().mean().item(),
            "primitive_acc": (primitive_prediction == primitive_labels).float().mean().item(),
        }
