from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn


@dataclass
class CurveStreamConfig:
    num_classes: int = 6
    dim: int = 32
    hidden: int = 64
    mem_long: int = 8
    mem_short: int = 4
    policy: Literal["curvature", "uniform", "recent"] = "curvature"


def curvature_scores(frames: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute discrete curvature scores for a trajectory.

    frames: [T, D]
    returns: [T] where edge positions are 0.

    We approximate curvature at time t via 1 - cos(angle(d_{t-1}, d_t)).
    """
    if frames.ndim != 2 or frames.size(0) < 3:
        return torch.zeros(frames.size(0), device=frames.device)

    d1 = frames[1:-1] - frames[:-2]
    d2 = frames[2:] - frames[1:-1]

    d1 = d1 / (d1.norm(dim=-1, keepdim=True) + eps)
    d2 = d2 / (d2.norm(dim=-1, keepdim=True) + eps)

    cos = (d1 * d2).sum(dim=-1).clamp(-1.0, 1.0)
    curv_mid = 1.0 - cos  # [T-2]

    out = torch.zeros(frames.size(0), device=frames.device)
    out[1:-1] = curv_mid
    return out


def select_memory_indices(
    frames: torch.Tensor, *, mem_long: int, mem_short: int, policy: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select frame indices to retain in (long_memory, short_memory).

    This is a *toy* approximation of CurveStream's "curvature-aware hierarchical memory" idea.

    - long_memory: keep top-K high-curvature frames (global, sparse).
    - short_memory: keep last-k frames (recent context).

    Returns:
      long_idx: [<=mem_long]
      short_idx: [<=mem_short]
    """
    t = frames.size(0)
    device = frames.device

    short_k = max(0, int(mem_short))
    short_idx = torch.arange(max(0, t - short_k), t, device=device, dtype=torch.long)

    if policy == "recent":
        return torch.empty(0, device=device, dtype=torch.long), short_idx

    if policy == "uniform":
        long_k = max(0, int(mem_long))
        if long_k == 0:
            return torch.empty(0, device=device, dtype=torch.long), short_idx
        # Uniformly sample (excluding the short tail).
        end = max(0, t - short_k)
        if end <= 0:
            return torch.empty(0, device=device, dtype=torch.long), short_idx
        idx = torch.linspace(0, end - 1, steps=min(long_k, end), device=device)
        long_idx = idx.round().to(torch.long).unique()
        return long_idx, short_idx

    # curvature
    long_k = max(0, int(mem_long))
    if long_k == 0 or t < 3:
        return torch.empty(0, device=device, dtype=torch.long), short_idx

    scores = curvature_scores(frames)

    # Do not pick from the short tail; mimic "recent frames" handled separately.
    if short_k > 0:
        scores[max(0, t - short_k) :] = -1.0

    topk = min(long_k, int((scores >= 0).sum().item()))
    if topk <= 0:
        return torch.empty(0, device=device, dtype=torch.long), short_idx

    long_idx = torch.topk(scores, k=topk, largest=True).indices.sort().values
    return long_idx, short_idx


class CurveStreamToyModel(nn.Module):
    def __init__(self, cfg: CurveStreamConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = nn.Sequential(
            nn.Linear(cfg.dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
        )
        self.attn = nn.MultiheadAttention(cfg.hidden, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.num_classes),
        )

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """frames: [B, T, D]"""
        b, t, d = frames.shape
        assert d == self.cfg.dim

        long_list = []
        short_list = []
        all_idx_list = []

        for i in range(b):
            long_idx, short_idx = select_memory_indices(
                frames[i], mem_long=self.cfg.mem_long, mem_short=self.cfg.mem_short, policy=self.cfg.policy
            )
            # Merge indices; keep order.
            all_idx = torch.cat([long_idx, short_idx]).unique(sorted=True)
            all_idx_list.append(all_idx)
            long_list.append(long_idx)
            short_list.append(short_idx)

        # Gather into a padded batch.
        max_m = max(int(x.numel()) for x in all_idx_list)
        mem = torch.zeros(b, max_m, d, device=frames.device)
        mem_mask = torch.ones(b, max_m, dtype=torch.bool, device=frames.device)
        for i, idx in enumerate(all_idx_list):
            m = int(idx.numel())
            if m == 0:
                continue
            mem[i, :m] = frames[i, idx]
            mem_mask[i, :m] = False

        mem_h = self.frame_encoder(mem)

        # Query with the last frame (streaming "current" context).
        q = self.frame_encoder(frames[:, -1:, :])

        attn_out, _ = self.attn(query=q, key=mem_h, value=mem_h, key_padding_mask=mem_mask)
        pooled = attn_out[:, 0, :]
        logits = self.classifier(pooled)

        stats = {
            "mem_tokens": torch.tensor([int(x.numel()) for x in all_idx_list], device=frames.device),
            "long_tokens": torch.tensor([int(x.numel()) for x in long_list], device=frames.device),
            "short_tokens": torch.tensor([int(x.numel()) for x in short_list], device=frames.device),
        }
        return logits, stats
