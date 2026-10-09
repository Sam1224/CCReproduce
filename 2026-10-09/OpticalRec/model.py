from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Mode = Literal["optical", "late_fusion"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_query_len: int
    max_meta_len: int
    d_model: int = 128
    img_patch: int = 16
    img_size: int = 224
    nhead: int = 4
    nlayers: int = 3
    dim_ff: int = 256
    dropout: float = 0.1


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = nn.Embedding(max(cfg.max_query_len, cfg.max_meta_len), cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.nlayers)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.tok(tokens) + self.pos(pos)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(pooled, dim=-1)


class VisionPatchEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(3, cfg.d_model, kernel_size=cfg.img_patch, stride=cfg.img_patch)
        num_patches = (cfg.img_size // cfg.img_patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.nlayers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch(x)  # [B, D, H', W']
        h = h.flatten(2).transpose(1, 2)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos[:, : h.size(1)]
        h = self.encoder(h)
        return F.normalize(self.norm(h[:, 0]), dim=-1)


class LateFusionProductEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.vision = VisionPatchEncoder(cfg)
        self.text = TextEncoder(cfg)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, images: torch.Tensor, meta_tokens: torch.Tensor, meta_masks: torch.Tensor) -> torch.Tensor:
        v = self.vision(images)
        t = self.text(meta_tokens, meta_masks)
        fused = self.proj(torch.cat([v, t], dim=-1))
        return F.normalize(fused, dim=-1)


class OpticalProductEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.card_vision = VisionPatchEncoder(cfg)
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        # In the paper, a frozen VLM further performs semantic-level processing over rendered inputs.
        # Here we approximate that second stage with a lightweight projection head on top of visual tokens.
        z = self.card_vision(cards)
        z = self.semantic_head(z)
        return F.normalize(z, dim=-1)


class FrozenVLMBackboneAdapter(nn.Module):
    """
    Placeholder for a heavier-weight faithful implementation.

    Expected full-paper behavior:
      1) render product image + metadata into a product card
      2) feed the rendered card into a frozen VLM backbone (e.g. Qwen3-VL)
      3) read out a homogeneous item embedding from the VLM hidden states

    This repository uses `OpticalProductEncoder` by default so the pipeline remains runnable.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *_args, **_kwargs) -> torch.Tensor:
        raise NotImplementedError("Swap in a real frozen VLM if heavyweight weights are available.")


class OpticalRecModel(nn.Module):
    def __init__(self, cfg: ModelConfig, mode: Mode = "optical") -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.query_encoder = TextEncoder(cfg)
        self.ctx_proj = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        if mode == "optical":
            self.item_encoder = OpticalProductEncoder(cfg)
        elif mode == "late_fusion":
            self.item_encoder = LateFusionProductEncoder(cfg)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def encode_items(self, item_bundle: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "optical":
            return self.item_encoder(item_bundle["cards"])
        return self.item_encoder(item_bundle["images"], item_bundle["meta_tokens"], item_bundle["meta_masks"])

    def encode_context(
        self,
        query_tokens: torch.Tensor,
        query_masks: torch.Tensor,
        history_ids: torch.Tensor,
        item_bundle: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        item_emb = self.encode_items(item_bundle)
        query_emb = self.query_encoder(query_tokens, query_masks)
        hist_emb = item_emb[history_ids].mean(dim=1)
        ctx = self.ctx_proj(torch.cat([query_emb, hist_emb], dim=-1))
        return F.normalize(ctx, dim=-1)

    def score_items(
        self,
        query_tokens: torch.Tensor,
        query_masks: torch.Tensor,
        history_ids: torch.Tensor,
        item_bundle: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ctx = self.encode_context(query_tokens, query_masks, history_ids, item_bundle)
        item_emb = self.encode_items(item_bundle)
        return ctx @ item_emb.T


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


@torch.no_grad()
def recall_at_k(scores: torch.Tensor, positives: torch.Tensor, k: int = 10) -> float:
    topk = scores.topk(k=k, dim=1).indices
    hit = (topk == positives.unsqueeze(1)).any(dim=1).float()
    return float(hit.mean().item())


@torch.no_grad()
def ndcg_at_k(scores: torch.Tensor, positives: torch.Tensor, k: int = 10) -> float:
    topk = scores.topk(k=k, dim=1).indices
    gains = (topk == positives.unsqueeze(1)).float()
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
    dcg = (gains * discounts.unsqueeze(0)).sum(dim=1)
    return float(dcg.mean().item())

