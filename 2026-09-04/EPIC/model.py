from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from data import CATALOG, ITEM_ID_TO_ITEM, NUM_ITEMS, CODEBOOK_SIZE, TOKEN_DIM


class MaskedSIDBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.item_embed = nn.Embedding(NUM_ITEMS, hidden_dim)
        self.category_embed = nn.Embedding(CODEBOOK_SIZE, hidden_dim)
        self.brand_embed = nn.Embedding(CODEBOOK_SIZE, hidden_dim)
        self.style_embed = nn.Embedding(CODEBOOK_SIZE, hidden_dim)
        self.sid_heads = nn.ModuleList([nn.Linear(hidden_dim, CODEBOOK_SIZE) for _ in range(TOKEN_DIM)])
        self.item_head = nn.Linear(hidden_dim, NUM_ITEMS)

    def encode_history(self, history_items: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embed(history_items)
        weights = torch.linspace(0.6, 1.0, steps=history_items.size(1), device=history_items.device)
        context = (embedded * weights.view(1, -1, 1)).mean(dim=1)
        return context

    def forward(self, history_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_history(history_items)
        sid_logits = torch.stack([head(context) for head in self.sid_heads], dim=1)
        item_logits = self.item_head(context)
        return sid_logits, item_logits


class EPICAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.item_embed = nn.Embedding(NUM_ITEMS, hidden_dim)
        self.transition_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def feasible_mask(self, partial_sid: List[int | None], device: torch.device) -> torch.Tensor:
        mask = torch.zeros(NUM_ITEMS, dtype=torch.bool, device=device)
        for item in CATALOG:
            keep = True
            for idx, token in enumerate(partial_sid):
                if token is None:
                    continue
                if item.sid[idx] != token:
                    keep = False
                    break
            if keep:
                mask[item.item_id] = True
        return mask

    def item_posterior(self, history_items: torch.Tensor, partial_sid: List[int | None], base_item_logits: torch.Tensor) -> torch.Tensor:
        device = history_items.device
        mask = self.feasible_mask(partial_sid, device=device)
        history_embed = self.item_embed(history_items)
        recent = history_embed[:, -3:, :].mean(dim=1)
        candidate_embed = self.item_embed.weight.unsqueeze(0).expand(history_items.size(0), -1, -1)
        recent_expand = recent.unsqueeze(1).expand_as(candidate_embed)
        transition_scores = self.transition_mlp(torch.cat([candidate_embed, recent_expand], dim=-1)).squeeze(-1)
        masked_logits = base_item_logits + transition_scores
        masked_logits = masked_logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        return torch.softmax(masked_logits, dim=-1)

    def marginalize_to_sid(self, posterior: torch.Tensor) -> torch.Tensor:
        sid_posteriors = []
        for position in range(TOKEN_DIM):
            logits = torch.zeros(posterior.size(0), CODEBOOK_SIZE, device=posterior.device)
            for item in CATALOG:
                logits[:, item.sid[position]] += posterior[:, item.item_id]
            sid_posteriors.append(logits)
        return torch.stack(sid_posteriors, dim=1)


def sid_cross_entropy(sid_logits: torch.Tensor, target_sid: torch.Tensor) -> torch.Tensor:
    losses = []
    for idx in range(TOKEN_DIM):
        losses.append(F.cross_entropy(sid_logits[:, idx, :], target_sid[:, idx]))
    return sum(losses) / len(losses)


def decode_with_epic(backbone: MaskedSIDBackbone, adapter: EPICAdapter, history_items: torch.Tensor, use_epic: bool) -> torch.Tensor:
    sid_logits, item_logits = backbone(history_items)
    batch_size = history_items.size(0)
    decoded = torch.full((batch_size, TOKEN_DIM), -1, dtype=torch.long, device=history_items.device)
    for position in range(TOKEN_DIM):
        fused_rows = []
        for batch_index in range(batch_size):
            base_row = sid_logits[batch_index, position, :]
            if use_epic:
                partial = [decoded[batch_index, idx].item() if decoded[batch_index, idx].item() >= 0 else None for idx in range(TOKEN_DIM)]
                posterior = adapter.item_posterior(
                    history_items[batch_index : batch_index + 1],
                    partial,
                    item_logits[batch_index : batch_index + 1],
                )
                sid_posterior = adapter.marginalize_to_sid(posterior)
                base_row = base_row + torch.log(sid_posterior[0, position, :] + 1e-8)
            fused_rows.append(base_row)
        fused = torch.stack(fused_rows, dim=0)
        decoded[:, position] = fused.argmax(dim=-1)
    return decoded


def rank_items_from_sid(decoded_sid: torch.Tensor, item_logits: torch.Tensor, adapter: EPICAdapter | None, history_items: torch.Tensor | None = None) -> torch.Tensor:
    batch_size = decoded_sid.size(0)
    scores = item_logits.clone()
    for batch_index in range(batch_size):
        sid = decoded_sid[batch_index].tolist()
        for item in CATALOG:
            matches = sum(int(item.sid[pos] == sid[pos]) for pos in range(TOKEN_DIM))
            scores[batch_index, item.item_id] += matches * 2.0
    if adapter is not None and history_items is not None:
        posterior = adapter.item_posterior(history_items, [None] * TOKEN_DIM, item_logits)
        scores = scores + posterior * 2.5
    return scores


def ranking_metrics(item_scores: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, int] = (5, 10)) -> Dict[str, float]:
    order = item_scores.argsort(dim=-1, descending=True)
    metrics: Dict[str, float] = {}
    for k in ks:
        topk = order[:, :k]
        hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
        metrics[f"recall@{k}"] = hits.mean().item()
        gains = []
        for row, target in zip(topk, targets):
            if target.item() in row.tolist():
                rank = row.tolist().index(target.item()) + 1
                gains.append(1.0 / torch.log2(torch.tensor(rank + 1, dtype=torch.float32)).item())
            else:
                gains.append(0.0)
        metrics[f"ndcg@{k}"] = sum(gains) / len(gains)
    return metrics
