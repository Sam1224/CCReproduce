from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RegRetConfig:
    region_dim: int = 32
    global_dim: int = 32
    text_dim: int = 32
    hidden_dim: int = 64
    emb_dim: int = 48
    num_attrs: int = 24
    temperature: float = 0.07


class RegionAwareEncoder(nn.Module):
    def __init__(self, cfg: RegRetConfig):
        super().__init__()
        self.region_proj = nn.Sequential(
            nn.Linear(cfg.region_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.global_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )
        self.query = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.key = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.value = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.out = nn.Sequential(
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, region_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        region = self.region_proj(region_feat)
        global_ctx = self.global_proj(global_feat)
        attn = torch.sigmoid(
            (self.query(region) * self.key(global_ctx)).sum(dim=-1, keepdim=True) / math.sqrt(region.size(-1))
        )
        selected = attn * self.value(global_ctx)
        fused = self.out(torch.cat([region, region + selected], dim=-1))
        return F.normalize(fused, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, cfg: RegRetConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.emb_dim),
        )

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(text_feat), dim=-1)


class RegRetToyModel(nn.Module):
    def __init__(self, cfg: RegRetConfig):
        super().__init__()
        self.cfg = cfg
        self.query_encoder = RegionAwareEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.query_head = nn.Linear(cfg.emb_dim, cfg.num_attrs)
        self.text_head = nn.Linear(cfg.emb_dim, cfg.num_attrs)

    def encode_query(self, region_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(region_feat, global_feat)

    def encode_text(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_feat)

    def region_logits(self, region_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        return self.query_head(self.encode_query(region_feat, global_feat))

    def text_logits(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.text_head(self.encode_text(text_feat))

    def similarity(self, region_feat: torch.Tensor, global_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        query = self.encode_query(region_feat, global_feat)
        text = self.encode_text(text_feat)
        return query @ text.t()
