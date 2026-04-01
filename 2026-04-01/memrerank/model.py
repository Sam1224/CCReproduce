from __future__ import annotations

import torch
import torch.nn as nn


class MemRerankToyModel(nn.Module):
    """Toy implementation of preference memory for reranking.

    Paper: MemRerank: Preference Memory for Personalized Product Reranking (arXiv:2603.29247)

    This is a simplified baseline:
    - build a query-independent memory vector from user history
    - score item = dot(query_repr + alpha*memory, item_emb)
    """

    def __init__(
        self,
        *,
        num_items: int = 5000,
        vocab_size: int = 2048,
        dim: int = 128,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items, dim)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.hist_attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def encode_query(self, query_token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_emb(query_token_ids).mean(dim=1)

    def encode_memory(self, history_item_ids: torch.Tensor) -> torch.Tensor:
        # history_item_ids: [B, H]
        h = self.item_emb(history_item_ids)  # [B,H,D]
        w = self.hist_attn(h).squeeze(-1)  # [B,H]
        w = torch.softmax(w, dim=-1)
        return (w.unsqueeze(-1) * h).sum(dim=1)

    def score(self, query: torch.Tensor, memory: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        item = self.item_emb(item_id)
        u = query + self.alpha * memory
        return (u * item).sum(dim=-1)

    def forward(
        self,
        *,
        history_item_ids: torch.Tensor,
        query_token_ids: torch.Tensor,
        pos_item_id: torch.Tensor,
        neg_item_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.encode_query(query_token_ids)
        m = self.encode_memory(history_item_ids)
        pos = self.score(q, m, pos_item_id)
        neg = self.score(q, m, neg_item_id)
        return pos, neg
