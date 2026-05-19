from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import VOCAB


class PatchImageEncoder(nn.Module):
    def __init__(self, patch_size: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens: (B, N, C)."""
        x = self.proj(images)  # (B,C,H',W')
        x = x.flatten(2).transpose(1, 2)  # (B,N,C)
        return self.ln(x)


class TextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), hidden_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(text_ids)


class TIGERFG(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        patch_size: int = 8,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.image_encoder = PatchImageEncoder(patch_size=patch_size, hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.post_ln = nn.LayerNorm(hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, embedding_dim)
        self.item_proj = nn.Linear(hidden_dim, embedding_dim)

    def encode_query(self, query_images: torch.Tensor) -> torch.Tensor:
        patches = self.image_encoder(query_images)
        pooled = patches.mean(dim=1)
        emb = F.normalize(self.query_proj(pooled), dim=-1)
        return emb

    def encode_item(
        self,
        item_images: torch.Tensor,
        item_text: torch.Tensor,
        gt_box_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        patches = self.image_encoder(item_images)  # (B,N,C)
        text_tokens = self.text_encoder(item_text)  # (B,L,C)
        text_vec = text_tokens.mean(dim=1, keepdim=True)  # (B,1,C)

        attn_out, attn_weights = self.cross_attn(query=text_vec, key=patches, value=patches, need_weights=True)
        # attn_weights: (B,1,N)
        fused = self.post_ln(attn_out + text_vec)  # (B,1,C)
        fused = fused.squeeze(1)
        item_emb = F.normalize(self.item_proj(fused), dim=-1)

        out: Dict[str, torch.Tensor] = {
            "item_emb": item_emb,
            "item_patches": patches,
            "patch_attn": attn_weights.squeeze(1),
        }

        if gt_box_mask is not None:
            weights = gt_box_mask.float()
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            teacher_hidden = torch.einsum("bn,bnc->bc", weights, patches)
            out["teacher_item_emb"] = F.normalize(self.item_proj(teacher_hidden), dim=-1)

        return out

    def forward(
        self,
        query_images: torch.Tensor,
        item_images: torch.Tensor,
        item_text: torch.Tensor,
        gt_box_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        query_emb = self.encode_query(query_images)
        item_out = self.encode_item(item_images=item_images, item_text=item_text, gt_box_mask=gt_box_mask)

        return {
            "query_emb": query_emb,
            **item_out,
        }
