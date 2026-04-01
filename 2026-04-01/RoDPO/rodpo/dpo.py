from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    logp_pos: torch.Tensor,
    logp_neg: torch.Tensor,
    logp_pos_ref: torch.Tensor,
    logp_neg_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    delta = (logp_pos - logp_neg) - (logp_pos_ref - logp_neg_ref)
    return -F.logsigmoid(beta * delta).mean()


@torch.no_grad()
def sample_negative_items(
    scores: torch.Tensor,
    pos_items: torch.Tensor,
    num_items: int,
    strategy: str = "topk",
    topk: int = 50,
) -> torch.Tensor:
    batch = scores.size(0)

    if strategy == "random":
        neg = torch.randint(0, num_items, (batch,), device=scores.device)
        neg = torch.where(neg == pos_items, (neg + 1) % num_items, neg)
        return neg

    masked = scores.clone()
    masked[torch.arange(batch, device=scores.device), pos_items] = -1e9

    if strategy == "hard":
        return torch.argmax(masked, dim=-1)

    if strategy == "topk":
        k = min(topk, num_items - 1)
        topk_items = torch.topk(masked, k=k, dim=-1).indices
        pick = torch.randint(0, k, (batch,), device=scores.device)
        return topk_items[torch.arange(batch, device=scores.device), pick]

    raise ValueError(f"unknown negative sampling strategy: {strategy}")
