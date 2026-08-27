from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VdaOutput:
    logits: torch.Tensor
    token_logits: torch.Tensor
    visual_dependence: torch.Tensor
    attention: torch.Tensor


class VDAModel(nn.Module):
    def __init__(self, vocab_size: int = 60, dim: int = 24, hidden: int = 64, classes: int = 3) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.region_proj = nn.Linear(dim, dim)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, classes))
        self.token_head = nn.Linear(dim, vocab_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> VdaOutput:
        token_emb = self.token_emb(batch["token_ids"])
        regions = self.region_proj(batch["image_regions"])
        attn = torch.softmax(torch.einsum("btd,brd->btr", token_emb, regions) / regions.shape[-1] ** 0.5, dim=-1)
        visual_ctx = torch.einsum("btr,brd->btd", attn, regions)
        vd = F.cosine_similarity(token_emb, visual_ctx, dim=-1).sigmoid()
        weighted_tokens = (token_emb * vd.unsqueeze(-1)).mean(dim=1)
        visual_summary = visual_ctx.mean(dim=1)
        logits = self.fusion(torch.cat([weighted_tokens, visual_summary], dim=-1))
        token_logits = self.token_head(token_emb + visual_ctx)
        return VdaOutput(logits=logits, token_logits=token_logits, visual_dependence=vd, attention=attn)


def visually_modulated_loss(output: VdaOutput, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_tokens = batch["token_ids"]
    ce = F.cross_entropy(output.token_logits.reshape(-1, output.token_logits.shape[-1]), target_tokens.reshape(-1), reduction="none")
    weights = 0.5 + batch["visual_dependence"].reshape(-1)
    return (ce * weights).mean()


def vc_ot_loss(current_vd: torch.Tensor, reference_vd: torch.Tensor) -> torch.Tensor:
    current = current_vd / current_vd.sum(dim=1, keepdim=True).clamp_min(1e-6)
    reference = reference_vd / reference_vd.sum(dim=1, keepdim=True).clamp_min(1e-6)
    strength = (current_vd.sum(dim=1) - reference_vd.sum(dim=1)).abs().mean()
    structure = F.mse_loss(current.cumsum(dim=1), reference.cumsum(dim=1))
    independent_mass = current[:, 3:].sum(dim=1).mean()
    return strength + structure + 0.1 * independent_mass
