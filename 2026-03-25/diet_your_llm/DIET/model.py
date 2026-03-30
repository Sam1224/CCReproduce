from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True means "blocked" for PyTorch MultiheadAttention attn_mask when using float mask.
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        dim_mask: Optional[torch.Tensor] = None,
        return_act: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # key_padding_mask: (B,T) bool where True means "pad".
        attn_mask = causal_mask(x.shape[1], x.device)

        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop(a)

        h = self.norm2(x)
        f = self.ff(h)
        x = x + self.drop(f)

        if dim_mask is not None:
            x = x * dim_mask.view(1, 1, -1)

        return x, (x.detach() if return_act else None)


@dataclass
class TinyLMConfig:
    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    nlayer: int = 2
    d_ff: int = 256
    max_len: int = 64
    dropout: float = 0.1


class TinyTransformerLM(nn.Module):
    """A tiny causal LM used to demonstrate DIET's dimension-wise pruning."""

    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TinyBlock(cfg.d_model, cfg.nhead, cfg.d_ff, dropout=cfg.dropout) for _ in range(cfg.nlayer)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.register_buffer("dim_mask", torch.ones(cfg.d_model), persistent=False)

    def set_dim_mask(self, mask: Optional[torch.Tensor]) -> None:
        if mask is None:
            self.dim_mask = torch.ones(self.cfg.d_model, device=self.dim_mask.device)
        else:
            if mask.ndim != 1 or mask.numel() != self.cfg.d_model:
                raise ValueError("mask must be shape (d_model,)")
            self.dim_mask = mask.to(self.dim_mask.device, dtype=self.dim_mask.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # input_ids: (B,T) ; attn_mask: (B,T) where True means real tokens.
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # Apply dimension mask at embedding level too (structured pruning proxy).
        if self.dim_mask is not None:
            x = x * self.dim_mask.view(1, 1, -1)

        key_padding_mask = ~attn_mask
        acts: Optional[List[torch.Tensor]] = [] if return_activations else None

        for blk in self.blocks:
            x, act = blk(x, key_padding_mask=key_padding_mask, dim_mask=self.dim_mask, return_act=return_activations)
            if return_activations and acts is not None and act is not None:
                acts.append(act)

        x = self.norm(x)

        # Next-token prediction head on the last *real* token.
        last_index = attn_mask.long().sum(dim=1).clamp_min(1) - 1
        last_h = x[torch.arange(bsz, device=device), last_index]
        logits = self.lm_head(last_h)
        return logits, acts


def lm_loss(logits: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target_id)
