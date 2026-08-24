from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).float().unsqueeze(-1)
        pooled = (self.embedding(tokens) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.projection(pooled), dim=-1)


class ResidualQuantizer(nn.Module):
    def __init__(self, hidden_size: int = 128, codebooks: int = 3, codebook_size: int = 16):
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(codebooks, codebook_size, hidden_size) * 0.02)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = embeddings
        quantized = torch.zeros_like(embeddings)
        codes = []
        for codebook in self.codebooks:
            distances = torch.cdist(residual.unsqueeze(1), codebook.unsqueeze(0)).squeeze(1)
            code_ids = distances.argmin(dim=-1)
            selected = codebook[code_ids]
            quantized = quantized + selected
            residual = residual - selected.detach()
            codes.append(code_ids)
        return quantized, torch.stack(codes, dim=-1)


def sid_reward(query_codes: torch.Tensor, item_codes: torch.Tensor) -> torch.Tensor:
    matches = query_codes.eq(item_codes).float()
    weights = torch.linspace(1.0, 0.4, matches.size(1), device=matches.device)
    return (matches * weights).sum(dim=-1) / weights.sum()


class SSRGRPOModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.encoder = MeanEncoder(vocab_size, hidden_size)
        self.quantizer = ResidualQuantizer(hidden_size)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dense = self.encoder(tokens)
        quantized, codes = self.quantizer(dense)
        return F.normalize(dense + 0.1 * quantized, dim=-1), codes

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        query, query_codes = self.encode(batch["query_tokens"])
        positive, positive_codes = self.encode(batch["positive_tokens"])
        negative, negative_codes = self.encode(batch["negative_tokens"])
        positive_dense = (query * positive).sum(dim=-1)
        negative_dense = (query * negative).sum(dim=-1)
        positive_sparse = sid_reward(query_codes, positive_codes)
        negative_sparse = sid_reward(query_codes, negative_codes)
        return {
            "positive_score": 0.8 * positive_sparse + 0.2 * positive_dense,
            "negative_score": 0.8 * negative_sparse + 0.2 * negative_dense,
            "dense_query": query,
            "dense_positive": positive,
            "dense_negative": negative,
        }


def ssr_loss(outputs: Dict[str, torch.Tensor], margin: float = 0.2) -> torch.Tensor:
    retrieval_dpo = -F.logsigmoid((outputs["positive_score"] - outputs["negative_score"]) / 0.1).mean()
    masked_grpo = F.relu(margin - outputs["positive_score"] + outputs["negative_score"]).mean()
    alignment = 1.0 - outputs["positive_score"].mean()
    return retrieval_dpo + masked_grpo + 0.1 * alignment
