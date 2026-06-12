from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_len: int
    d_model: int = 192
    nhead: int = 6
    nlayers: int = 3
    dim_ff: int = 512
    dropout: float = 0.1


class KAEEncoder(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        slot_sizes: List[int],
        pad_id: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.slot_sizes = slot_sizes

        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.nlayers)

        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, s) for s in slot_sizes])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, L], attn_mask: [B, L] (1 for valid)
        bsz, seqlen = x.shape
        pos = torch.arange(seqlen, device=x.device).unsqueeze(0).expand(bsz, -1)

        h = self.tok(x) + self.pos(pos)
        key_padding_mask = attn_mask == 0
        h = self.enc(h, src_key_padding_mask=key_padding_mask)

        # mean pool over valid tokens
        denom = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom

        return [head(pooled) for head in self.heads]

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        forced_slot: Optional[int] = None,
        forced_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        logits = self.forward(x, attn_mask)
        pred = torch.stack([lg.argmax(dim=-1) for lg in logits], dim=1)

        if forced_slot is not None and forced_token_id is not None:
            pred[:, forced_slot] = int(forced_token_id)

        return pred
