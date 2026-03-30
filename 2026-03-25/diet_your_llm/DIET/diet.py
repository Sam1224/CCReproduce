from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from model import TinyTransformerLM


@dataclass
class DietConfig:
    sparsity: float = 0.2
    samples_per_task: int = 100


def profile_dim_importance(
    model: TinyTransformerLM,
    batches: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """Activation-magnitude profiling (paper: use activation magnitudes).

    Returns:
        importance: (d_model,) float tensor
    """

    model.eval()
    d_model = model.cfg.d_model

    total = torch.zeros(d_model, device=device)
    denom = 0

    with torch.no_grad():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            _, acts = model(input_ids=input_ids, attn_mask=attn_mask, return_activations=True)
            if not acts:
                continue

            # Average |activation| across layers and tokens.
            layer_scores = []
            for a in acts:
                # a: (B,T,D)
                m = attn_mask.unsqueeze(-1).type_as(a)
                denom_bt = m.sum().clamp_min(1.0)
                layer_scores.append((a.abs() * m).sum(dim=(0, 1)) / denom_bt)

            score = torch.stack(layer_scores, dim=0).mean(dim=0)
            total += score
            denom += 1

    if denom == 0:
        return total
    return total / float(denom)


def task_keep_mask_from_importance(importance: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Return a boolean keep-mask for top-(1-sparsity) dims."""
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0,1)")

    d = int(importance.numel())
    k_keep = max(1, int(round((1.0 - sparsity) * d)))
    _, idx = torch.topk(importance, k=k_keep, largest=True)

    keep = torch.zeros(d, dtype=torch.bool, device=importance.device)
    keep[idx] = True
    return keep


def majority_vote_keep_mask(task_keeps: List[torch.Tensor]) -> torch.Tensor:
    """Majority vote to build a single global keep mask.

    DIET uses majority voting across tasks to produce a global mask.

    Tie-breaking: keep (safer).
    """

    if not task_keeps:
        raise ValueError("task_keeps is empty")

    stacked = torch.stack([t.to(dtype=torch.long) for t in task_keeps], dim=0)  # (N,D)
    votes = stacked.sum(dim=0)
    n = stacked.shape[0]
    return votes * 2 >= n


def apply_pruning_mask(model: TinyTransformerLM, keep_mask: torch.Tensor) -> None:
    model.set_dim_mask(keep_mask.to(dtype=model.dim_mask.dtype))


def summarize_mask(mask: torch.Tensor) -> Tuple[int, float]:
    keep = int(mask.to(dtype=torch.long).sum().item())
    sparsity = 1.0 - keep / float(mask.numel())
    return keep, sparsity
