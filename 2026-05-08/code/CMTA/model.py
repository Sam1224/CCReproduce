"""CMTA model: dual-branch (coarse GRU + fine Transformer) over CLIP visual / text features.

In the paper, BLIP produces a per-frame caption and CLIP encodes both image and caption.
Here we expose stub `BlipCaptioner` / `ClipEncoder` interfaces so users can drop in real
HuggingFace pipelines without touching the rest of the model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CMTAConfig:
    feature_dim: int = 64
    seq_len: int = 16
    gru_hidden: int = 128
    fine_layers: int = 2
    fine_heads: int = 4
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Pluggable feature stubs (replace with real BLIP / CLIP at integration time).
# ---------------------------------------------------------------------------
class BlipCaptioner(nn.Module):
    """Stub. In production: HuggingFace `Salesforce/blip-image-captioning-large`."""

    def forward(self, frames: torch.Tensor) -> list[str]:  # noqa: ARG002
        # TODO[paper]: produce per-frame natural-language captions using BLIP.
        raise NotImplementedError("Replace with HuggingFace BLIP at integration time.")


class ClipEncoder(nn.Module):
    """Stub. In production: HuggingFace `openai/clip-vit-base-patch16` for both modalities."""

    def encode_image(self, frames: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        raise NotImplementedError("Replace with HuggingFace CLIP image encoder.")

    def encode_text(self, captions: list[str]) -> torch.Tensor:  # noqa: ARG002
        raise NotImplementedError("Replace with HuggingFace CLIP text encoder.")


# ---------------------------------------------------------------------------
# CMTA classifier
# ---------------------------------------------------------------------------
class CMTA(nn.Module):
    def __init__(self, cfg: CMTAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Coarse branch: GRU over per-frame cosine alignment values.
        self.coarse_gru = nn.GRU(input_size=1, hidden_size=cfg.gru_hidden, batch_first=True, num_layers=1)
        # Fine branch: Transformer over concatenated (visual, text) features.
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.feature_dim * 2,
            nhead=cfg.fine_heads,
            dim_feedforward=4 * cfg.feature_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fine_encoder = nn.TransformerEncoder(layer, num_layers=cfg.fine_layers)
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.gru_hidden + cfg.feature_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def cross_modal_alignment(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Cosine similarity per frame. Returns [B, T, 1]."""
        v_n = F.normalize(v, dim=-1)
        c_n = F.normalize(c, dim=-1)
        return (v_n * c_n).sum(dim=-1, keepdim=True)

    def forward(self, v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Coarse: temporal stability of cross-modal alignment.
        s = self.cross_modal_alignment(v, c)  # [B, T, 1]
        coarse_h, _ = self.coarse_gru(s)
        coarse_repr = coarse_h[:, -1, :]
        # Fine: inter-frame interactions of (visual, text).
        fused = torch.cat([v, c], dim=-1)  # [B, T, 2D]
        fine_repr = self.fine_encoder(fused).mean(dim=1)
        # Fusion -> binary logit
        logit = self.classifier(torch.cat([coarse_repr, fine_repr], dim=-1)).squeeze(-1)
        return logit


__all__ = ["CMTA", "CMTAConfig", "BlipCaptioner", "ClipEncoder"]
