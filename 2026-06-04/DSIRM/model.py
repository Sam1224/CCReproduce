from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass
class TokenizedBatch:
    token_ids: torch.Tensor  # (B, T)
    attn_mask: torch.Tensor  # (B, T)


class SimpleTokenizer:
    """A tiny whitespace tokenizer for toy reproduction.

    This keeps the code runnable without external model downloads.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}

    def build_vocab(self, texts: Iterable[str], min_freq: int = 1) -> None:
        freq: Dict[str, int] = {}
        for text in texts:
            for tok in text.lower().split():
                freq[tok] = freq.get(tok, 0) + 1

        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

    def encode_batch(self, texts: List[str], max_len: int = 24) -> TokenizedBatch:
        ids: List[List[int]] = []
        masks: List[List[int]] = []
        for t in texts:
            toks = t.lower().split()[:max_len]
            row = [self.vocab.get(tok, 1) for tok in toks]
            mask = [1] * len(row)
            while len(row) < max_len:
                row.append(0)
                mask.append(0)
            ids.append(row)
            masks.append(mask)

        return TokenizedBatch(
            token_ids=torch.tensor(ids, dtype=torch.long),
            attn_mask=torch.tensor(masks, dtype=torch.float32),
        )


class MeanPoolEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(token_ids)  # (B, T, H)
        attn_mask = attn_mask.unsqueeze(-1)  # (B, T, 1)
        x = x * attn_mask
        pooled = x.sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(self.proj(pooled))
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled


class DualEncoder(nn.Module):
    """Siamese dual encoder (shared weights), like a production embedding backbone."""

    def __init__(self, vocab_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.encoder = MeanPoolEncoder(vocab_size=vocab_size, hidden_size=hidden_size)

    def encode(self, token_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(token_ids=token_ids, attn_mask=attn_mask)

    def forward(self, query_batch: TokenizedBatch, item_batch: TokenizedBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.encode(query_batch.token_ids, query_batch.attn_mask)
        d = self.encode(item_batch.token_ids, item_batch.attn_mask)
        return q, d


class QuerySidPredictor(nn.Module):
    """Query -> hierarchical SID predictor.

    Paper: uses a generative LLM. Here we use a lightweight multi-head predictor to keep
    the toy reproduction runnable.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_codebooks: int, codebook_size: int) -> None:
        super().__init__()
        self.encoder = MeanPoolEncoder(vocab_size=vocab_size, hidden_size=hidden_size)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, codebook_size) for _ in range(num_codebooks)])

    def forward(self, query_batch: TokenizedBatch) -> torch.Tensor:
        # returns logits: (B, num_codebooks, K)
        h = self.encoder(query_batch.token_ids, query_batch.attn_mask)  # (B, H)
        logits = torch.stack([head(h) for head in self.heads], dim=1)
        return logits
