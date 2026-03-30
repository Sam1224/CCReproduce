from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_graph(docs: torch.Tensor, top_m: int = 3) -> torch.Tensor:
    """Build a soft adjacency matrix by cosine similarity."""
    docs_n = F.normalize(docs, dim=-1)
    sim = docs_n @ docs_n.t()  # (K,K)
    K = sim.shape[0]
    sim.fill_diagonal_(-1e9)

    # keep top-m neighbors
    vals, idx = torch.topk(sim, k=min(top_m, K - 1), dim=-1)
    adj = torch.zeros_like(sim)
    adj.scatter_(1, idx, vals)
    adj = torch.softmax(adj, dim=-1)
    return adj


class GraphReranker(nn.Module):
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.combine = nn.Sequential(
            nn.Linear(2, 1),
        )

    def forward(self, q: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
        # Base relevance
        q_rep = q.unsqueeze(0).expand(docs.shape[0], -1)
        base = self.scorer(torch.cat([q_rep, docs], dim=-1)).squeeze(-1)  # (K,)

        # Graph enrichment: propagate base scores over doc graph
        adj = build_graph(docs)
        neigh = adj @ base.unsqueeze(-1)
        neigh = neigh.squeeze(-1)

        feat = torch.stack([base, neigh], dim=-1)
        return self.combine(feat).squeeze(-1)
