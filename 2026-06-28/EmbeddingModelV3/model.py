from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


def mean_pool(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """emb: [B, T, D], mask: [B, T] (1 for valid)."""
    mask_f = mask.float().unsqueeze(-1)
    summed = (emb * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / denom


@dataclass
class Batch:
    q_ids: torch.Tensor
    d_ids: torch.Tensor
    q_mask: torch.Tensor
    d_mask: torch.Tensor


class DualEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.d_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def encode_query(self, q_ids: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(q_ids)
        x = mean_pool(x, q_mask)
        x = self.norm(self.q_proj(x))
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def encode_doc(self, d_ids: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(d_ids)
        x = mean_pool(x, d_mask)
        x = self.norm(self.d_proj(x))
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def forward(self, batch: Batch) -> torch.Tensor:
        q = self.encode_query(batch.q_ids, batch.q_mask)
        d = self.encode_doc(batch.d_ids, batch.d_mask)
        return (q * d).sum(dim=-1)


def pad_2d(seqs: Sequence[Sequence[int]], pad_id: int = 0) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def make_mask(ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    return (ids != pad_id).to(torch.long)


def collate_pair_batch(batch_items: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    q_ids = pad_2d([it["q_ids"] for it in batch_items], pad_id=pad_id)
    d_ids = pad_2d([it["d_ids"] for it in batch_items], pad_id=pad_id)
    return {
        "q_ids": q_ids,
        "d_ids": d_ids,
        "q_mask": make_mask(q_ids, pad_id=pad_id),
        "d_mask": make_mask(d_ids, pad_id=pad_id),
        "label": torch.tensor([it["label"] for it in batch_items], dtype=torch.float32),
        "grade": torch.tensor([it.get("grade", 0) for it in batch_items], dtype=torch.long),
        "bucket": [it.get("bucket", "") for it in batch_items],
    }
