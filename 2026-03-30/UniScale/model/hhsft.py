from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class HHSFTConfig:
    vocab_size: int = 4096
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    domains: int = 3
    segment_count: int = 3  # user/query/item


class DomainGatedFFN(nn.Module):
    """A tiny 2-expert FFN with a domain-conditioned gate.

    This is a toy approximation of heterogeneity-aware routing.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, domains: int, dropout: float) -> None:
        super().__init__()
        self.expert_a = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.expert_b = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = nn.Embedding(domains, 1)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), domain_id: (B,)
        gate = torch.sigmoid(self.gate(domain_id)).unsqueeze(1)  # (B, 1, 1)
        return gate * self.expert_a(x) + (1.0 - gate) * self.expert_b(x)


class HHSFTBlock(nn.Module):
    def __init__(self, cfg: HHSFTConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = DomainGatedFFN(cfg.embed_dim, hidden_dim=cfg.embed_dim * 4, domains=cfg.domains, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-norm attention
        h = self.ln1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        h2 = self.ln2(x)
        f = self.ffn(h2, domain_id)
        x = x + self.dropout(f)
        return x


class HHSFTModel(nn.Module):
    def __init__(self, cfg: HHSFTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(512, cfg.embed_dim)
        self.segment_emb = nn.Embedding(cfg.segment_count, cfg.embed_dim)
        self.domain_emb = nn.Embedding(cfg.domains, cfg.embed_dim)

        self.blocks = nn.ModuleList([HHSFTBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.embed_dim)

        self.click_head = nn.Linear(cfg.embed_dim, 1)
        self.purchase_head = nn.Linear(cfg.embed_dim, 1)

    def _build_sequence(
        self,
        user_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate segments and return (x, segment_ids)."""

        bsz = user_tokens.size(0)
        device = user_tokens.device

        seg_user = torch.zeros((bsz, user_tokens.size(1)), dtype=torch.long, device=device)
        seg_query = torch.ones((bsz, query_tokens.size(1)), dtype=torch.long, device=device)
        seg_item = torch.full((bsz, item_tokens.size(1)), 2, dtype=torch.long, device=device)

        tokens = torch.cat([user_tokens, query_tokens, item_tokens], dim=1)  # (B, T)
        seg_ids = torch.cat([seg_user, seg_query, seg_item], dim=1)  # (B, T)

        pos = torch.arange(tokens.size(1), device=device).unsqueeze(0).expand(bsz, -1)

        x = self.token_emb(tokens) + self.pos_emb(pos) + self.segment_emb(seg_ids) + self.domain_emb(domain_id).unsqueeze(1)
        return x, seg_ids

    def _hierarchical_attn_mask(self, seg_ids: torch.Tensor) -> torch.Tensor:
        """Toy 'hierarchical' attention mask.

        Allow all tokens to attend to user+query, but restrict item tokens from attending to other item tokens.
        This is a cheap approximation of hierarchical fusion.

        Returns a float mask in MultiheadAttention format: (T, T) or (B, T, T).
        We'll build (B, T, T) with -inf for masked positions.
        """

        # seg_ids: (B, T)
        bsz, t = seg_ids.shape
        device = seg_ids.device

        item = seg_ids == 2
        mask = torch.zeros((bsz, t, t), device=device)

        # For each batch: if query is item token and key is item token, mask it.
        item_q = item.unsqueeze(2)  # (B, T, 1)
        item_k = item.unsqueeze(1)  # (B, 1, T)
        item_item = item_q & item_k
        mask[item_item] = float("-inf")

        # keep self attention allowed
        diag = torch.eye(t, device=device).bool().unsqueeze(0)
        mask[diag] = 0.0

        return mask

    def forward(
        self,
        user_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, seg_ids = self._build_sequence(user_tokens, query_tokens, item_tokens, domain_id)
        attn_mask = self._hierarchical_attn_mask(seg_ids)

        for blk in self.blocks:
            x = blk(x, domain_id=domain_id, attn_mask=attn_mask)

        x = self.final_ln(x)

        # Pool: take mean of query segment as a proxy for ranking score
        query_mask = (seg_ids == 1).unsqueeze(-1)
        pooled = (x * query_mask).sum(dim=1) / query_mask.sum(dim=1).clamp(min=1)

        click_logit = self.click_head(pooled).squeeze(-1)
        purchase_logit = self.purchase_head(pooled).squeeze(-1)
        return click_logit, purchase_logit
