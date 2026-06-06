from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class RandomProjectionANNIndex:
    """A minimal ANN index based on random projection.

    Approximation is controlled by `proj_dim`. Smaller proj_dim => lower recall.

    This is a toy index used to reproduce the *metric behavior* discussed in the
    paper, not to compete with HNSW/FAISS.
    """

    proj: torch.Tensor  # [D, d]
    base_embeddings: torch.Tensor  # [N, D]
    base_labels: torch.Tensor  # [N]
    base_proj: torch.Tensor  # [N, d]

    @staticmethod
    def build(
        base_embeddings: torch.Tensor,
        base_labels: torch.Tensor,
        proj_dim: int,
        seed: int,
    ) -> "RandomProjectionANNIndex":
        if base_embeddings.ndim != 2:
            raise ValueError("base_embeddings must be [N, D]")

        torch.manual_seed(seed)
        dim = base_embeddings.shape[1]
        proj = torch.randn(dim, proj_dim)
        proj = torch.nn.functional.normalize(proj, dim=0)

        base_proj = base_embeddings @ proj
        return RandomProjectionANNIndex(
            proj=proj,
            base_embeddings=base_embeddings,
            base_labels=base_labels,
            base_proj=base_proj,
        )

    def search(
        self, query_embeddings: torch.Tensor, k: int, oversample: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, distances) for approximate top-k neighbors.

        We first retrieve `k * oversample` candidates in the projected space,
        then re-rank them by the true distance in the original space.

        Distances returned are computed in the **original** embedding space.
        """

        query_proj = query_embeddings @ self.proj
        # Use L2 distance in projected space for candidate ranking.
        # [Q, N]
        d2_proj = (
            query_proj.pow(2).sum(dim=1, keepdim=True)
            + self.base_proj.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * (query_proj @ self.base_proj.t())
        )

        candidate_k = min(self.base_proj.shape[0], k * oversample)
        candidate_idx = torch.topk(d2_proj, k=candidate_k, largest=False).indices  # [Q, C]

        q = query_embeddings.unsqueeze(1)  # [Q, 1, D]
        cand = self.base_embeddings[candidate_idx]  # [Q, C, D]
        d2 = (q - cand).pow(2).sum(dim=-1)  # [Q, C]

        topk = torch.topk(d2, k=k, largest=False)
        approx_idx = candidate_idx.gather(1, topk.indices)  # [Q, k]
        return approx_idx, torch.sqrt(torch.clamp(topk.values, min=0.0))


def exact_knn(
    base_embeddings: torch.Tensor, query_embeddings: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute-force exact kNN in original space."""

    q2 = query_embeddings.pow(2).sum(dim=1, keepdim=True)
    b2 = base_embeddings.pow(2).sum(dim=1).unsqueeze(0)
    d2 = q2 + b2 - 2.0 * (query_embeddings @ base_embeddings.t())
    idx = torch.topk(d2, k=k, largest=False).indices
    dist = torch.sqrt(torch.clamp(d2.gather(1, idx), min=0.0))
    return idx, dist
