from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BiEncoderConfig:
    vocab_size: int = 4096
    d_model: int = 128
    max_len: int = 20
    dropout: float = 0.1


class MeanEncoder(nn.Module):
    def __init__(self, cfg: BiEncoderConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        mask = (token_ids != 0).to(embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)


class DenseRetriever(nn.Module):
    def __init__(self, cfg: BiEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.query_encoder = MeanEncoder(cfg)
        self.doc_encoder = MeanEncoder(cfg)

    def encode_query(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(token_ids), dim=-1)

    def encode_doc(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.doc_encoder(token_ids), dim=-1)

    def score_pairs(self, query_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        query_vec = self.encode_query(query_ids)
        doc_vec = self.encode_doc(doc_ids)
        return (query_vec * doc_vec).sum(dim=-1)

    def score_matrix(self, query_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        query_vec = self.encode_query(query_ids)
        doc_vec = self.encode_doc(doc_ids)
        return query_vec @ doc_vec.T
