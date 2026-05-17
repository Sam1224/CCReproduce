from __future__ import annotations

import torch
import torch.nn as nn


class MeanPoolTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T], mask: [B, T] bool
        emb = self.embedding(token_ids)  # [B, T, D]
        mask_f = mask.float().unsqueeze(-1)
        summed = (emb * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom
