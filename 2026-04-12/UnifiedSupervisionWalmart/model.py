from __future__ import annotations

from typing import List

import torch
from torch import nn


class TokenMeanEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        x = self.emb(token_ids).mean(dim=1)
        return torch.tanh(self.proj(x))


class BiEncoderRetriever(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64):
        super().__init__()
        self.query_encoder = TokenMeanEncoder(vocab_size=vocab_size, dim=dim)
        self.item_encoder = TokenMeanEncoder(vocab_size=vocab_size, dim=dim)

    def encode_query(self, q: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(q)

    def encode_item(self, d: torch.Tensor) -> torch.Tensor:
        return self.item_encoder(d)

    def score(self, q_vec: torch.Tensor, d_vec: torch.Tensor) -> torch.Tensor:
        q = torch.nn.functional.normalize(q_vec, dim=-1)
        d = torch.nn.functional.normalize(d_vec, dim=-1)
        return (q * d).sum(dim=-1)
