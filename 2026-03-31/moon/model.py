from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedMoE(nn.Module):
    """A tiny guided MoE FFN.

    token_types: 0=other, 1=category, 2=attribute.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.expert_other = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.expert_cat = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.expert_attr = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        # x: (L,d)
        x = self.ln(x)
        out = torch.empty_like(x)

        mask_other = token_types == 0
        mask_cat = token_types == 1
        mask_attr = token_types == 2

        if mask_other.any():
            out[mask_other] = x[mask_other] + self.expert_other(x[mask_other])
        if mask_cat.any():
            out[mask_cat] = x[mask_cat] + self.expert_cat(x[mask_cat])
        if mask_attr.any():
            out[mask_attr] = x[mask_attr] + self.expert_attr(x[mask_attr])
        return out


class MoonEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, d_img: int = 64) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.moe = GuidedMoE(d_model)

        self.img_proj = nn.Linear(d_img, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def encode_text(self, token_ids: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        x = self.tok(token_ids)
        x = self.moe(x, token_types)
        return x.mean(dim=0)

    def encode_images(self, images_full: torch.Tensor, images_core: torch.Tensor) -> torch.Tensor:
        # images: (M, d_img)
        full = self.img_proj(images_full).mean(dim=0)
        core = self.img_proj(images_core).mean(dim=0)
        return (full + core) / 2

    def encode_product(
        self,
        token_ids: torch.Tensor,
        token_types: torch.Tensor,
        images_full: torch.Tensor,
        images_core: torch.Tensor,
    ) -> torch.Tensor:
        text = self.encode_text(token_ids, token_types)
        img = self.encode_images(images_full, images_core)
        z = self.fuse(torch.cat([text, img], dim=-1))
        return self.out_ln(z)


def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = (q @ k.t()) / temperature
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)
