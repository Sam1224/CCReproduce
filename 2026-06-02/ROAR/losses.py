from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class ROARBatch:
    # group_sizes[i] = g_i
    group_sizes: List[int]


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (N, D), b: (M, D)
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return a @ b.t()


def contrastive_loss_with_false_negative_mask(
    q: torch.Tensor,
    d: torch.Tensor,
    temperature: float = 0.07,
    margin_delta: float = 0.1,
) -> torch.Tensor:
    """Stage-1 InfoNCE-style contrastive loss.

    Implements the paper's false-negative margin masking:

        m_ij = 1[ cos(q_i, d_j) <= cos(q_i, d_i) + delta ]

    Any in-batch product d_j whose similarity to q_i exceeds that of the labeled
    positive (d_i) by more than delta is excluded from the denominator.

    Notes:
    - This toy implementation uses only in-batch documents as negatives.
    - The original Qwen3 recipe includes multiple negative pools; those can be
      added later without changing the core interface.
    """

    sim = cosine_sim_matrix(q, d)  # (B, B)
    pos = sim.diag().unsqueeze(1)  # (B, 1)

    # keep negatives where sim <= pos + delta
    keep_mask = sim <= (pos + margin_delta)

    # Always keep the positive itself
    keep_mask.fill_diagonal_(True)

    logits = sim / temperature
    logits = logits.masked_fill(~keep_mask, -1e9)

    targets = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, targets)


def roar_alignment_loss(
    q: torch.Tensor,
    d: torch.Tensor,
    batch: ROARBatch,
    tau_align: float = 0.07,
) -> torch.Tensor:
    """Stage-2 ROAR alignment loss.

    Follows the paper (Eq. 2-6) in a flattened-group batching scheme.

    We assume the doc batch is a concatenation of per-query groups:
        [p_0,1, ..., p_0,g0, p_1,1, ..., p_1,g1, ...]

    and group_sizes provides each g_i.

    Alignment loss:
        L_align = -(1/C) * sum_i sum_{k=0}^{g_i-2} log sigma(DeltaOR_{i,k})

    where DeltaOR compares consecutive items inside the group's ranking.
    """

    sim = cosine_sim_matrix(q, d)  # (N, M)
    probs = F.softmax(sim / tau_align, dim=1)

    offsets: List[int] = []
    s = 0
    for g in batch.group_sizes:
        offsets.append(s)
        s += g

    # Build DeltaOR terms
    deltas: List[torch.Tensor] = []
    for i, g in enumerate(batch.group_sizes):
        base = offsets[i]
        for k in range(g - 1):
            p_hi = probs[i, base + k]
            p_lo = probs[i, base + k + 1]
            odds_hi = p_hi / (1.0 - p_hi + 1e-8)
            odds_lo = p_lo / (1.0 - p_lo + 1e-8)
            delta_or = torch.log(odds_hi + 1e-8) - torch.log(odds_lo + 1e-8)
            deltas.append(delta_or)

    if not deltas:
        return torch.tensor(0.0, device=q.device)

    delta_tensor = torch.stack(deltas, dim=0)
    return -torch.log(torch.sigmoid(delta_tensor) + 1e-8).mean()
