from __future__ import annotations

import torch
import torch.nn as nn


class RAIRModel(nn.Module):
    def __init__(
        self,
        n_cat: int = 40,
        n_brand: int = 120,
        n_color: int = 20,
        d: int = 64,
        use_rules: bool = True,
    ) -> None:
        super().__init__()
        self.use_rules = use_rules

        self.emb_cat = nn.Embedding(n_cat, d)
        self.emb_brand = nn.Embedding(n_brand, d)
        self.emb_color = nn.Embedding(n_color, d)

        # rule embedding: {0,1}^3
        self.rule_emb = nn.Embedding(2, d)

        # q: (cat, brand, color)=3*d; item: (cat, brand, color_text, color_img)=4*d
        in_dim = d * 7 + (d * 3 if use_rules else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    def forward(
        self,
        q_cat: torch.Tensor,
        q_brand: torch.Tensor,
        q_color: torch.Tensor,
        item_cat: torch.Tensor,
        item_brand: torch.Tensor,
        item_color_text: torch.Tensor,
        item_color_img: torch.Tensor,
        rule_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = torch.cat(
            [
                self.emb_cat(q_cat),
                self.emb_brand(q_brand),
                self.emb_color(q_color),
            ],
            dim=-1,
        )
        item = torch.cat(
            [
                self.emb_cat(item_cat),
                self.emb_brand(item_brand),
                self.emb_color(item_color_text),
                self.emb_color(item_color_img),
            ],
            dim=-1,
        )

        if self.use_rules:
            rules = self.rule_emb(rule_ids).view(rule_ids.shape[0], -1)
            feat = torch.cat([q, item, rules], dim=-1)
        else:
            feat = torch.cat([q, item], dim=-1)
        return self.mlp(feat)
