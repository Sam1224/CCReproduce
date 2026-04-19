from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenMerger(nn.Module):
    def __init__(self, embed_dim: int, merge_ratio: int = 2):
        super().__init__()
        self.merge_ratio = merge_ratio
        self.proj = nn.Linear(embed_dim * merge_ratio, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        bsz, seq_len, dim = x.shape
        r = self.merge_ratio
        if seq_len < r:
            return x
        new_len = seq_len // r
        x = x[:, : new_len * r, :]
        x = x.reshape(bsz, new_len, dim * r)
        return self.proj(x)


class CausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        max_len: int = 2048,
        merge_ratio: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.merger = TokenMerger(embed_dim=embed_dim, merge_ratio=merge_ratio)
        self.merge_ratio = merge_ratio

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_len: int,
        apply_token_merger: bool = True,
    ) -> torch.Tensor:
        # input_ids: [B, T]
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        tok = self.embed(input_ids)
        pos = self.pos(torch.arange(seq_len, device=device)).unsqueeze(0)
        x = tok + pos

        if apply_token_merger and prefix_len > 0:
            prefix = x[:, :prefix_len, :]
            suffix = x[:, prefix_len:, :]
            merged_prefix = self.merger(prefix)
            x = torch.cat([merged_prefix, suffix], dim=1)

            prefix_mask = attention_mask[:, :prefix_len]
            suffix_mask = attention_mask[:, prefix_len:]
            merged_prefix_mask = prefix_mask[:, : merged_prefix.shape[1] * self.merge_ratio : self.merge_ratio]
            attention_mask = torch.cat([merged_prefix_mask, suffix_mask], dim=1)

        t = x.shape[1]
        causal = torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)

        x = self.blocks(x, mask=causal, src_key_padding_mask=~attention_mask)
        return self.lm_head(x)
