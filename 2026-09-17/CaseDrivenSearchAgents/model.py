from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 128
    max_len: int = 32


class MeanEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        mask = (ids != 0).to(x.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return self.ln(pooled)


class AllInOneRelevanceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q_enc = MeanEncoder(cfg)
        self.d_enc = MeanEncoder(cfg)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.retrieval_head = nn.Linear(cfg.d_model, 1)
        self.coarse_head = nn.Linear(cfg.d_model, 4)
        self.fine_head = nn.Linear(cfg.d_model, 1)

    def forward(self, q_ids: torch.Tensor, d_ids: torch.Tensor):
        q = self.q_enc(q_ids)
        d = self.d_enc(d_ids)
        feat = torch.cat([q, d, q * d], dim=-1)
        h = self.fuse(feat)
        return {
            "retrieval_logit": self.retrieval_head(h).squeeze(-1),
            "coarse_logits": self.coarse_head(h),
            "fine_score": self.fine_head(h).squeeze(-1),
        }
