from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]

    @property
    def pad_id(self) -> int:
        return 0

    @staticmethod
    def build(texts: Iterable[str], *, min_freq: int = 1) -> "Vocab":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        token_to_id: Dict[str, int] = {"<pad>": 0}
        for token, count in sorted(counts.items()):
            if count >= min_freq:
                token_to_id[token] = len(token_to_id)

        return Vocab(token_to_id=token_to_id)


def tokenize(text: str) -> List[str]:
    return [t.strip(".,:;!?()\"'“”").lower() for t in text.split() if t.strip()]


def encode(text: str, vocab: Vocab, *, max_len: int = 96) -> torch.Tensor:
    ids = [vocab.token_to_id.get(t, vocab.pad_id) for t in tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab.pad_id] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


class BiEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        emb = self.embedding(input_ids)  # [B, L, D]
        mask = (input_ids != 0).float().unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return torch.nn.functional.normalize(self.proj(pooled), dim=-1)


def batch_encode(texts: Sequence[str], vocab: Vocab, *, max_len: int = 96) -> torch.Tensor:
    return torch.stack([encode(t, vocab, max_len=max_len) for t in texts], dim=0)
