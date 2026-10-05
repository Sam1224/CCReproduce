from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyKVModel(nn.Module):
    def __init__(self, vocab_size: int = 128, dim: int = 64):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, vocab_size)

    def prefill(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.token(tokens)
        return hidden, self.key(hidden), self.value(hidden)

    def decode(self, query_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        query = self.query(self.token(query_tokens)).unsqueeze(1)
        scale = keys.shape[-1] ** -0.5
        logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) * scale
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights.unsqueeze(1), values).squeeze(1)
        return self.out(context)

    def forward(self, batch: Dict[str, torch.Tensor], mask: torch.Tensor = None) -> torch.Tensor:
        _, keys, values = self.prefill(batch["tokens"])
        return self.decode(batch["query"], keys, values, mask)


def proxy_scores(tokens: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    token_change = F.pad((tokens[:, 1:] != tokens[:, :-1]).float(), (1, 0), value=1.0)
    key_norm = keys.norm(dim=-1)
    sink_bonus = torch.zeros_like(key_norm)
    sink_bonus[:, :2] = 1.0
    return 0.55 * key_norm / key_norm.mean(dim=1, keepdim=True).clamp_min(1e-6) + 0.35 * token_change + 0.10 * sink_bonus


def selective_reconstruction_scores(keys: torch.Tensor, values: torch.Tensor, selected_query_idx: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = keys.shape
    gather_idx = selected_query_idx.unsqueeze(-1).expand(-1, -1, keys.shape[-1])
    selected_queries = torch.gather(keys, dim=1, index=gather_idx)
    attention = torch.softmax(torch.matmul(selected_queries, keys.transpose(1, 2)) / (keys.shape[-1] ** 0.5), dim=-1)
    contribution = attention * values.norm(dim=-1).unsqueeze(1)
    return contribution.mean(dim=1)


def kv2_mask(tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, budget_ratio: float = 0.2, query_ratio: float = 0.15, refinement_steps: int = 1) -> torch.Tensor:
    batch_size, seq_len, _ = keys.shape
    budget = max(1, int(seq_len * budget_ratio))
    query_count = max(1, int(seq_len * query_ratio))
    scores = proxy_scores(tokens, keys)
    for _ in range(refinement_steps):
        selected = scores.topk(query_count, dim=1).indices
        reconstruction = selective_reconstruction_scores(keys, values, selected)
        scores = 0.35 * scores + 0.65 * reconstruction
    keep = scores.topk(budget, dim=1).indices
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=keys.device)
    mask.scatter_(1, keep, True)
    return mask
