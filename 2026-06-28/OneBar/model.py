from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class QueryGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.tr = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt_inp: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_e = self.pos(self.emb(src))
        tgt_e = self.pos(self.emb(tgt_inp))
        # causal mask for decoder
        T = tgt_inp.size(1)
        causal = torch.triu(torch.ones(T, T, device=tgt_inp.device), diagonal=1).bool()
        out = self.tr(
            src=src_e,
            tgt=tgt_e,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=causal,
        )
        return self.lm_head(out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 12,
    ) -> torch.Tensor:
        self.eval()
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            tgt_pad = ys
            tgt_mask = tgt_pad.eq(0)
            logits = self.forward(src, tgt_pad, src_key_padding_mask, tgt_mask)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if torch.all(next_id.squeeze(-1).eq(eos_id)):
                break
        return ys


def pad_2d(seqs: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    m = max(len(s) for s in seqs)
    out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out
