from __future__ import annotations

import torch


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return a @ b.T


def recall_at_k(sim: torch.Tensor, k: int) -> float:
    """Compute Recall@k for a square similarity matrix where the diagonal is the positive pair."""
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"sim must be NxN, got {tuple(sim.shape)}")

    n = sim.shape[0]
    _, topk = torch.topk(sim, k=k, dim=1)
    targets = torch.arange(n, device=sim.device)
    hits = (topk == targets[:, None]).any(dim=1).float().mean().item()
    return float(hits)
