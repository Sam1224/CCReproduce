from __future__ import annotations

from dataclasses import dataclass

import torch

from model import SemanticIdRecommender


@dataclass(frozen=True)
class CapeConfig:
    beam_size: int = 50
    expansion_size: int = 25
    top_k: int = 10
    lambda_mem: float = 0.35
    head_cutoff: int = 48


def minmax(values: torch.Tensor) -> torch.Tensor:
    low = values.min(dim=-1, keepdim=True).values
    high = values.max(dim=-1, keepdim=True).values
    return (values - low) / (high - low).clamp_min(1e-8)


def cooccurrence_support(history: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """S_u(c) = sum_{j in H_u} U[j, c]."""

    rows = memory[history]
    return rows.sum(dim=1)


@torch.no_grad()
def cape_rank(
    model: SemanticIdRecommender,
    history: torch.Tensor,
    memory: torch.Tensor,
    config: CapeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top item IDs and calibrated scores for each user."""

    batch_size, _ = history.shape
    log_probs = model.score_all(history)
    beam_scores, beam_items = torch.topk(log_probs, k=config.beam_size, dim=-1)
    support = cooccurrence_support(history, memory)

    tail_mask = torch.zeros_like(support, dtype=torch.bool)
    tail_mask[:, config.head_cutoff :] = True
    in_beam = torch.zeros_like(tail_mask)
    in_beam.scatter_(1, beam_items, True)
    expansion_scores = support.masked_fill(~tail_mask | in_beam, -1.0)
    _, expanded_items = torch.topk(expansion_scores, k=config.expansion_size, dim=-1)

    candidates = torch.cat([beam_items, expanded_items], dim=-1)
    candidate_log_probs = log_probs.gather(1, candidates)
    candidate_support = support.gather(1, candidates)

    tail_candidate = candidates >= config.head_cutoff
    evidence = candidate_support.masked_fill(~tail_candidate, 0.0)
    calibrated = minmax(candidate_log_probs) + config.lambda_mem * minmax(evidence)

    # Top-1 anchoring: keep the backbone's first item in the most prominent slot.
    top1 = beam_items[:, :1]
    top1_mask = candidates == top1
    calibrated = calibrated.masked_fill(top1_mask, calibrated.max(dim=-1, keepdim=True).values + 1.0)

    scores, order = torch.topk(calibrated, k=config.top_k, dim=-1)
    ranked_items = candidates.gather(1, order)

    # Remove accidental duplicates while preserving order. If duplicates occur,
    # backfill with backbone items. This mirrors production post-processing.
    deduped = []
    for row in range(batch_size):
        seen: set[int] = set()
        values: list[int] = []
        for item in ranked_items[row].tolist() + beam_items[row].tolist():
            if item not in seen:
                values.append(item)
                seen.add(item)
            if len(values) == config.top_k:
                break
        deduped.append(values)
    return torch.tensor(deduped, device=history.device, dtype=torch.long), scores

