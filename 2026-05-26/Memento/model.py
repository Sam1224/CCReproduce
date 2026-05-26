from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class CTRModel(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_vec: torch.Tensor, hist_agg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([query_vec, hist_agg], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class HistoryAggConfig:
    k: int
    mode: str  # lastn | mmr
    lambda_relevance: float = 0.6


def aggregate_history_lastn(history_vecs: torch.Tensor, k: int) -> torch.Tensor:
    # history_vecs: (B, H, D)
    if k <= 0:
        return torch.zeros(history_vecs.shape[0], history_vecs.shape[2], device=history_vecs.device)
    k = min(k, history_vecs.shape[1])
    return history_vecs[:, -k:, :].mean(dim=1)


def aggregate_history_mmr(
    *,
    history_vecs: torch.Tensor,
    query_vec: torch.Tensor,
    k: int,
    lambda_relevance: float,
) -> torch.Tensor:
    """MMR aggregation; selection happens on CPU (numpy) for clarity."""

    from .retrieval import mmr_select

    batch_size, hist_len, dim = history_vecs.shape
    out = torch.zeros(batch_size, dim, device=history_vecs.device)

    history_np = history_vecs.detach().cpu().numpy()
    query_np = query_vec.detach().cpu().numpy()

    for b in range(batch_size):
        idx = mmr_select(query=query_np[b], candidates=history_np[b], k=k, lambda_relevance=lambda_relevance)
        if len(idx) == 0:
            continue
        out[b] = history_vecs[b, idx, :].mean(dim=0)

    return out
