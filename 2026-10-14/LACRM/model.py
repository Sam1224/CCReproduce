from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


VIEW_NAMES = [
    "direct_homogeneous",
    "intra_brand_structural",
    "intra_brand_cross_channel",
    "cross_category_substitution",
    "temporal_campaign",
]

# field order: category, brand, intent, style, audience, price, channel, tier
VIEW_FIELD_MASKS = torch.tensor(
    [
        [1, 1, 1, 0, 0, 1, 0, 0],  # direct homogeneous
        [1, 1, 0, 0, 0, 0, 1, 1],  # intra-brand structural
        [1, 1, 0, 0, 0, 0, 1, 1],  # intra-brand cross-channel
        [0, 0, 1, 1, 1, 1, 0, 0],  # cross-category substitution
        [0, 0, 1, 0, 1, 0, 0, 0],  # temporal campaign
    ],
    dtype=torch.float32,
)


@dataclass
class LACRMConfig:
    n_categories: int = 8
    n_brands: int = 24
    n_intents: int = 10
    n_styles: int = 6
    n_audiences: int = 6
    n_prices: int = 5
    n_channels: int = 4
    n_tiers: int = 4
    embed_dim: int = 32
    proj_dim: int = 48
    rerank_hidden: int = 96
    behavior_dim: int = 12


class DualProfileEncoder(nn.Module):
    """A PyTorch surrogate for the paper's AR / QR + asymmetric embedding stage.

    The original paper uses:
      - Qwen3-4B-Instruct (+ SFT / GRPO) to generate profiles
      - Qwen3-Embedding-8B to embed AR / QR without further embedding training

    Here we keep the *same interfaces* but replace them with a lightweight,
    learnable field encoder so the whole stack stays runnable in a toy setup.
    """

    def __init__(self, cfg: LACRMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.category_emb = nn.Embedding(cfg.n_categories, d)
        self.brand_emb = nn.Embedding(cfg.n_brands, d)
        self.intent_emb = nn.Embedding(cfg.n_intents, d)
        self.style_emb = nn.Embedding(cfg.n_styles, d)
        self.audience_emb = nn.Embedding(cfg.n_audiences, d)
        self.price_emb = nn.Embedding(cfg.n_prices, d)
        self.channel_emb = nn.Embedding(cfg.n_channels, d)
        self.tier_emb = nn.Embedding(cfg.n_tiers, d)

        self.view_emb = nn.Embedding(len(VIEW_NAMES), d)
        self.source_role = nn.Parameter(torch.randn(d) * 0.02)
        self.target_role = nn.Parameter(torch.randn(d) * 0.02)

        self.query_mlp = nn.Sequential(
            nn.Linear(d, cfg.proj_dim),
            nn.GELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
        )
        self.anchor_mlp = nn.Sequential(
            nn.Linear(d, cfg.proj_dim),
            nn.GELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
        )

    def field_stack(self, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [
                self.category_emb(fields["category"]),
                self.brand_emb(fields["brand"]),
                self.intent_emb(fields["intent"]),
                self.style_emb(fields["style"]),
                self.audience_emb(fields["audience"]),
                self.price_emb(fields["price"]),
                self.channel_emb(fields["channel"]),
                self.tier_emb(fields["tier"]),
            ],
            dim=1,
        )  # [N, 8, D]

    def encode_anchors(self, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        stack = self.field_stack(fields)
        base = stack.mean(dim=1) + self.target_role
        anchor = self.anchor_mlp(base)
        return F.normalize(anchor, dim=-1)

    def encode_queries(
        self,
        fields: dict[str, torch.Tensor],
        view_ids: torch.Tensor,
    ) -> torch.Tensor:
        stack = self.field_stack(fields)  # [N,8,D]
        masks = VIEW_FIELD_MASKS.to(stack.device)[view_ids]      # [N,8]
        masked = stack * masks.unsqueeze(-1)
        denom = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = masked.sum(dim=1) / denom
        pooled = pooled + self.view_emb(view_ids) + self.source_role
        query = self.query_mlp(pooled)
        return F.normalize(query, dim=-1)


class LambdaRankReranker(nn.Module):
    """A differentiable LambdaRank-style scorer.

    Original paper uses LambdaMART. Because this repo requires a PyTorch
    implementation, we keep the same feature contract but optimize a neural
    list-wise scorer with lambda-style pairwise supervision.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

