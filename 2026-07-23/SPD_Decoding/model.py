from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Catalog


class GenerativeSlateModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_items: int,
        slate_size: int,
    ) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, hidden_dim)
        self.user_encoder = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, num_items)
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))
        self.slate_size = slate_size

    def encode_user(self, user_state: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(torch.cat([user_state, constraints], dim=-1))

    def forward(self, user_state: torch.Tensor, constraints: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = user_state.size(0)
        encoded_user = self.encode_user(user_state, constraints)
        prev_emb = self.start_token.unsqueeze(0).expand(batch_size, -1)
        hidden = encoded_user.unsqueeze(0)
        logits_per_step = []

        for step in range(self.slate_size):
            output, hidden = self.decoder(prev_emb.unsqueeze(1), hidden)
            step_hidden = output.squeeze(1) + encoded_user
            step_logits = self.output_proj(step_hidden)
            logits_per_step.append(step_logits)
            teacher_token = targets[:, step]
            prev_emb = self.item_embeddings(teacher_token)

        return torch.stack(logits_per_step, dim=1)


@dataclass
class DecodeResult:
    slates: torch.Tensor
    diagnostics: Dict[str, float]


@torch.no_grad()
def decode_greedy(
    model: GenerativeSlateModel,
    user_state: torch.Tensor,
    constraints: torch.Tensor,
) -> DecodeResult:
    batch_size = user_state.size(0)
    encoded_user = model.encode_user(user_state, constraints)
    prev_emb = model.start_token.unsqueeze(0).expand(batch_size, -1)
    hidden = encoded_user.unsqueeze(0)
    chosen = []

    for _ in range(model.slate_size):
        output, hidden = model.decoder(prev_emb.unsqueeze(1), hidden)
        step_hidden = output.squeeze(1) + encoded_user
        logits = model.output_proj(step_hidden)
        if chosen:
            mask = F.one_hot(torch.stack(chosen, dim=1), num_classes=logits.size(-1)).sum(dim=1).bool()
            logits = logits.masked_fill(mask, -1e9)
        next_item = logits.argmax(dim=-1)
        chosen.append(next_item)
        prev_emb = model.item_embeddings(next_item)

    slates = torch.stack(chosen, dim=1)
    return DecodeResult(slates=slates, diagnostics={"avg_lambda_tail": 0.0, "avg_lambda_diversity": 0.0})


@torch.no_grad()
def stochastic_primal_dual_decode(
    model: GenerativeSlateModel,
    user_state: torch.Tensor,
    constraints: torch.Tensor,
    catalog: Catalog,
    step_size: float = 0.85,
    temperature: float = 0.8,
) -> DecodeResult:
    batch_size = user_state.size(0)
    encoded_user = model.encode_user(user_state, constraints)
    prev_emb = model.start_token.unsqueeze(0).expand(batch_size, -1)
    hidden = encoded_user.unsqueeze(0)
    chosen = []
    lambdas = torch.ones(batch_size, 2, device=user_state.device)

    for _ in range(model.slate_size):
        output, hidden = model.decoder(prev_emb.unsqueeze(1), hidden)
        step_hidden = output.squeeze(1) + encoded_user
        logits = model.output_proj(step_hidden)

        if chosen:
            mask = F.one_hot(torch.stack(chosen, dim=1), num_classes=logits.size(-1)).sum(dim=1).bool()
            logits = logits.masked_fill(mask, -1e9)

        tail_signal = catalog.creator_groups.to(user_state.device).float().unsqueeze(0).expand(batch_size, -1)
        business_signal = catalog.business_scores.to(user_state.device).unsqueeze(0).expand(batch_size, -1)
        category_ids = catalog.item_categories.to(user_state.device)

        diversity_signal = torch.ones_like(logits)
        if chosen:
            chosen_tensor = torch.stack(chosen, dim=1)
            seen_categories = category_ids[chosen_tensor]
            for batch_index in range(batch_size):
                present = torch.unique(seen_categories[batch_index])
                diversity_signal[batch_index] = (~torch.isin(category_ids, present)).float()

        adjusted = logits
        adjusted = adjusted + lambdas[:, 0].unsqueeze(1) * (tail_signal - constraints[:, 0].unsqueeze(1))
        adjusted = adjusted + lambdas[:, 1].unsqueeze(1) * (diversity_signal - 0.5)
        adjusted = adjusted + 0.2 * business_signal
        adjusted = adjusted + 0.05 * torch.randn_like(adjusted)

        probs = torch.softmax(adjusted / temperature, dim=-1)
        next_item = torch.multinomial(probs, num_samples=1).squeeze(1)
        chosen.append(next_item)
        prev_emb = model.item_embeddings(next_item)

        chosen_tensor = torch.stack(chosen, dim=1)
        current_tail_ratio = catalog.creator_groups.to(user_state.device)[chosen_tensor].float().mean(dim=1)
        category_counts = []
        for batch_index in range(batch_size):
            unique_count = torch.unique(category_ids[chosen_tensor[batch_index]]).numel()
            category_counts.append(float(unique_count))
        category_counts = torch.tensor(category_counts, dtype=torch.float32, device=user_state.device)

        violations = torch.stack(
            [
                torch.relu(constraints[:, 0] - current_tail_ratio),
                torch.relu(constraints[:, 1] - category_counts),
            ],
            dim=1,
        )
        lambdas = torch.relu(lambdas + step_size * violations)

    diagnostics = {
        "avg_lambda_tail": float(lambdas[:, 0].mean().item()),
        "avg_lambda_diversity": float(lambdas[:, 1].mean().item()),
    }
    return DecodeResult(slates=torch.stack(chosen, dim=1), diagnostics=diagnostics)


def training_loss(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()
    return loss, {"token_accuracy": accuracy}
