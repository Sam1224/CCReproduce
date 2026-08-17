from __future__ import annotations

import torch
from torch import nn

from data import SessionMemory, candidate_to_vector, retrieve_candidates, turn_to_bow


class ConstraintAwareRanker(nn.Module):
    def __init__(self, query_dim: int, candidate_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        query_hidden = self.query_encoder(query_features)
        candidate_hidden = self.candidate_encoder(candidate_features)
        return self.scorer(torch.cat([query_hidden, candidate_hidden], dim=-1)).squeeze(-1)


class MACSPipeline:
    def __init__(self, ranker: ConstraintAwareRanker, catalog: list) -> None:
        self.ranker = ranker
        self.catalog = catalog
        self.memory = SessionMemory()

    def reset(self) -> None:
        self.memory = SessionMemory()

    @torch.no_grad()
    def recommend(self, turn, top_k: int = 3):
        self.memory.update(turn)
        candidates = retrieve_candidates(self.catalog, self.memory, top_k=max(8, top_k))
        if not candidates:
            return []
        query_features = torch.cat([turn_to_bow(turn), self.memory.as_vector()]).unsqueeze(0)
        ranked = []
        for candidate in candidates:
            candidate_features = candidate_to_vector(candidate).unsqueeze(0)
            score = self.ranker(query_features, candidate_features).item()
            ranked.append((score, candidate))
        ranked.sort(key=lambda row: row[0], reverse=True)
        return [candidate for _, candidate in ranked[:top_k]]
