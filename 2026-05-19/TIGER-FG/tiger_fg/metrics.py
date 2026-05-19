from __future__ import annotations

import torch


def recall_at_k(similarity: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """similarity: (num_queries, num_items), higher is better."""
    topk = torch.topk(similarity, k=k, dim=1).indices
    hit = (topk == targets[:, None]).any(dim=1).float().mean()
    return float(hit)
