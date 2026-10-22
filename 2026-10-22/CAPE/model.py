from __future__ import annotations

import torch
from torch import nn


class SemanticIdRecommender(nn.Module):
    """A small generative recommender proxy.

    The paper evaluates CAPE on generative recommenders that decode semantic item
    identifiers. For a compact reproduction, each item ID is treated as a complete
    semantic identifier. `score_all` is the teacher-forcing likelihood proxy used
    both for beam candidates and CAPE-added candidates.
    """

    def __init__(self, num_items: int, embedding_dim: int = 96, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embedding(history)
        _, hidden = self.encoder(embedded)
        return self.output(hidden[-1])

    @torch.no_grad()
    def score_all(self, history: torch.Tensor) -> torch.Tensor:
        logits = self(history)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def beam_candidates(self, history: torch.Tensor, beam_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_all(history)
        return torch.topk(scores, k=beam_size, dim=-1)

