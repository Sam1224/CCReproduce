from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    num_items: int
    num_categories: int
    seq_len: int
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 1
    keep_recent: int = 12
    merge_chunk: int = 12


class SequenceBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.num_items + 1, cfg.d_model)
        self.cat_emb = nn.Embedding(cfg.num_categories, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len + 8, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.attn = nn.Linear(cfg.d_model, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def _encode_embeddings(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        x = x + self.pos_emb(pos)
        hidden = self.encoder(x, src_key_padding_mask=~mask)
        logits = self.attn(hidden).masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = (hidden * weights).sum(dim=1)
        return pooled

    def _cat_embed(self, ad_cat: torch.Tensor) -> torch.Tensor:
        return self.cat_emb(ad_cat)


class FullAttentionTeacher(SequenceBackbone):
    def forward(self, seq: torch.Tensor, mask: torch.Tensor, ad_cat: torch.Tensor) -> torch.Tensor:
        token_emb = self.item_emb(seq)
        seq_repr = self._encode_embeddings(token_emb, mask)
        ad_repr = self._cat_embed(ad_cat)
        return self.classifier(torch.cat([seq_repr, ad_repr], dim=-1)).squeeze(-1)


class TruncatedBaseline(SequenceBackbone):
    def forward(self, seq: torch.Tensor, mask: torch.Tensor, ad_cat: torch.Tensor) -> torch.Tensor:
        trunc = max(4, self.cfg.keep_recent // 2)
        seq = seq[:, -trunc:]
        mask = mask[:, -trunc:]
        token_emb = self.item_emb(seq)
        seq_repr = self._encode_embeddings(token_emb, mask)
        ad_repr = self._cat_embed(ad_cat)
        return self.classifier(torch.cat([seq_repr, ad_repr], dim=-1)).squeeze(-1)


class TM20KStudent(SequenceBackbone):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.merge_proj = nn.Linear(cfg.d_model * 3, cfg.d_model)

    def _merge_one(self, seq: torch.Tensor) -> Tuple[torch.Tensor, int]:
        keep_recent = min(self.cfg.keep_recent, int(seq.shape[0]))
        older = seq[:-keep_recent]
        recent = seq[-keep_recent:]
        merged: List[torch.Tensor] = []

        if older.numel() > 0:
            for chunk in older.split(self.cfg.merge_chunk):
                emb = self.item_emb(chunk)
                litm = emb.mean(dim=0)
                uniq, counts = torch.unique(chunk, return_counts=True)
                weights = counts.to(emb.dtype)
                token_weight = torch.zeros(chunk.shape[0], dtype=emb.dtype, device=emb.device)
                for u, c in zip(uniq, weights):
                    token_weight = token_weight + (chunk == u).to(emb.dtype) * c
                token_weight = token_weight / token_weight.sum().clamp_min(1.0)
                patm = (emb * token_weight.unsqueeze(-1)).sum(dim=0)
                lptm = emb[-1]
                merged.append(self.merge_proj(torch.cat([litm, patm, lptm], dim=-1)))

        recent_emb = self.item_emb(recent)
        if merged:
            full = torch.cat([torch.stack(merged, dim=0), recent_emb], dim=0)
        else:
            full = recent_emb
        return full, int(full.shape[0])

    def _merge_batch(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        merged_batch: List[torch.Tensor] = []
        lengths: List[int] = []
        for row in seq:
            merged, length = self._merge_one(row)
            merged_batch.append(merged)
            lengths.append(length)
        max_len = max(lengths)
        padded = torch.zeros(seq.shape[0], max_len, self.cfg.d_model, device=seq.device)
        mask = torch.zeros(seq.shape[0], max_len, dtype=torch.bool, device=seq.device)
        for idx, (merged, length) in enumerate(zip(merged_batch, lengths)):
            padded[idx, :length] = merged
            mask[idx, :length] = True
        return padded, mask

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, ad_cat: torch.Tensor) -> torch.Tensor:
        del mask
        merged, merged_mask = self._merge_batch(seq)
        seq_repr = self._encode_embeddings(merged, merged_mask)
        ad_repr = self._cat_embed(ad_cat)
        return self.classifier(torch.cat([seq_repr, ad_repr], dim=-1)).squeeze(-1)
