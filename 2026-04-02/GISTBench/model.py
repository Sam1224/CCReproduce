from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GISTOutput:
    logits: torch.Tensor


class InterestExtractor(nn.Module):
    """A small proxy model for user-interest extraction.

    The paper evaluates LLMs on extracting user interests from interaction
    histories. Here we provide a compact learnable baseline that maps interaction
    sequences to a distribution over coarse interest categories.
    """

    def __init__(
        self,
        *,
        num_leaf_categories: int,
        num_interests: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.leaf_embedding = nn.Embedding(num_leaf_categories, hidden_size)
        self.signal_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.head = nn.Linear(hidden_size, num_interests)

    def forward(self, leaf_ids: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> GISTOutput:
        embedded = self.leaf_embedding(leaf_ids)
        weight_emb = self.signal_mlp(weights.unsqueeze(-1))
        encoded = embedded + weight_emb

        encoded = encoded * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = pooled / denom

        return GISTOutput(logits=self.head(pooled))
