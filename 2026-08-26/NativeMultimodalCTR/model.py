from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    item_repr: torch.Tensor
    gate: torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NativeMultimodalCTR(nn.Module):
    """Small native multimodal encoder plus CTR head."""

    def __init__(
        self,
        *,
        num_users: int,
        num_categories: int,
        text_dim: int,
        image_dim: int,
        embed_dim: int = 32,
        cat_dim: int = 8,
        hidden: int = 96,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.cat_emb = nn.Embedding(num_categories, cat_dim)
        self.text_proj = MLP(text_dim, hidden, embed_dim)
        self.image_proj = MLP(image_dim, hidden, embed_dim)
        self.fusion_gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
        ctr_in = embed_dim * 3 + cat_dim + 1
        self.ctr_head = nn.Sequential(
            nn.Linear(ctr_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def encode_item(self, text: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_z = F.normalize(self.text_proj(text), dim=-1)
        image_z = F.normalize(self.image_proj(image), dim=-1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([text_z, image_z], dim=-1)))
        item_z = F.normalize(gate * text_z + (1.0 - gate) * image_z, dim=-1)
        return item_z, gate.squeeze(-1)

    def triplet_loss(
        self,
        anchor_text: torch.Tensor,
        anchor_image: torch.Tensor,
        positive_text: torch.Tensor,
        positive_image: torch.Tensor,
        negative_text: torch.Tensor,
        negative_image: torch.Tensor,
        margin: float = 0.25,
    ) -> torch.Tensor:
        anchor, _ = self.encode_item(anchor_text, anchor_image)
        positive, _ = self.encode_item(positive_text, positive_image)
        negative, _ = self.encode_item(negative_text, negative_image)
        pos_dist = 1.0 - (anchor * positive).sum(dim=-1)
        neg_dist = 1.0 - (anchor * negative).sum(dim=-1)
        return F.relu(pos_dist - neg_dist + margin).mean()

    def forward(
        self,
        *,
        user_id: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        category: torch.Tensor,
        price: torch.Tensor,
    ) -> ModelOutput:
        user = self.user_emb(user_id)
        item, gate = self.encode_item(text, image)
        cat = self.cat_emb(category)
        feat = torch.cat([user, item, user * item, cat, price.unsqueeze(-1)], dim=-1)
        logits = self.ctr_head(feat).squeeze(-1)
        return ModelOutput(logits=logits, item_repr=item, gate=gate)
