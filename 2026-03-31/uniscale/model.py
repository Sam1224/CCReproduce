from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class DomainRoutedExpertFusion(nn.Module):
    """Sparse-ish MoE: route by domain embedding + soft gates."""

    def __init__(self, d: int, n_domains: int, n_experts: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.domain_emb = nn.Embedding(n_domains, d)
        self.router = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_experts))
        self.experts = nn.ModuleList([MLP(d, 4 * d, d, dropout=dropout) for _ in range(n_experts)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        # x: (B,d)
        d_emb = self.domain_emb(domain_id)
        gates = F.softmax(self.router(d_emb), dim=-1)  # (B,E)
        out = 0.0
        h = self.ln(x)
        for i, exp in enumerate(self.experts):
            out = out + gates[:, i : i + 1] * exp(h)
        return x + out


class DomainAwareGatedAttention(nn.Module):
    """Inject cross-domain interest into search representation."""

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.ln = nn.LayerNorm(d)

    def forward(self, query: torch.Tensor, interest: torch.Tensor) -> torch.Tensor:
        # query: (B,1,d), interest: (B,1,d)
        q = self.ln(query)
        k = self.ln(interest)
        a, _ = self.attn(q, k, k, need_weights=False)
        g = self.gate(torch.cat([query, a], dim=-1))
        return query + g * a


@dataclass
class TokenSpec:
    user: int = 0
    ctx: int = 1
    item: int = 2
    domain: int = 3
    interest: int = 4


class HeterogeneousTokenizer(nn.Module):
    def __init__(
        self,
        *,
        d_user: int,
        d_ctx: int,
        d_item: int,
        d_interest: int,
        d_model: int,
        n_domains: int,
    ) -> None:
        super().__init__()
        self.user = nn.Linear(d_user, d_model)
        self.ctx = nn.Linear(d_ctx, d_model)
        self.item = nn.Linear(d_item, d_model)
        self.interest = nn.Linear(d_interest, d_model)
        self.domain = nn.Embedding(n_domains, d_model)

        self.type_emb = nn.Embedding(8, d_model)
        self.spec = TokenSpec()

    def forward(
        self,
        *,
        user_dense: torch.Tensor,
        ctx_dense: torch.Tensor,
        item_dense: torch.Tensor,
        user_interest_dense: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return token sequence (B,T,d) and type ids (B,T)
        u = self.user(user_dense)
        c = self.ctx(ctx_dense)
        it = self.item(item_dense)
        intr = self.interest(user_interest_dense)
        dom = self.domain(domain_id)

        tokens = torch.stack([u, c, it, dom, intr], dim=1)
        type_ids = torch.tensor(
            [self.spec.user, self.spec.ctx, self.spec.item, self.spec.domain, self.spec.interest],
            device=tokens.device,
            dtype=torch.long,
        ).unsqueeze(0).repeat(tokens.shape[0], 1)

        tokens = tokens + self.type_emb(type_ids)
        return tokens, type_ids


class HHSFT(nn.Module):
    def __init__(
        self,
        *,
        d_user: int = 32,
        d_ctx: int = 16,
        d_item: int = 32,
        d_interest: int = 16,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_domains: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = HeterogeneousTokenizer(
            d_user=d_user,
            d_ctx=d_ctx,
            d_item=d_item,
            d_interest=d_interest,
            d_model=d_model,
            n_domains=n_domains,
        )

        self.intra_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.cross_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])

        self.interest_moe = DomainRoutedExpertFusion(d_model, n_domains, n_experts=4, dropout=dropout)
        self.fusion = DomainAwareGatedAttention(d_model, n_heads=n_heads, dropout=dropout)

        self.out_ln = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        *,
        user_dense: torch.Tensor,
        ctx_dense: torch.Tensor,
        item_dense: torch.Tensor,
        user_interest_dense: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        tokens, type_ids = self.tokenizer(
            user_dense=user_dense,
            ctx_dense=ctx_dense,
            item_dense=item_dense,
            user_interest_dense=user_interest_dense,
            domain_id=domain_id,
        )

        # Hierarchical interaction: intra-group (simulate by masking attention to same-type)
        for blk in self.intra_blocks:
            tokens = blk(tokens)

        # Entire-space user-interest fusion
        # Apply MoE only to interest token.
        interest_idx = 4
        interest = tokens[:, interest_idx, :]
        interest = self.interest_moe(interest, domain_id)
        tokens = torch.cat([tokens[:, :interest_idx, :], interest.unsqueeze(1), tokens[:, interest_idx + 1 :, :]], dim=1)

        # Cross-group interaction (full attention)
        for blk in self.cross_blocks:
            tokens = blk(tokens)

        # Domain-aware injection: let (user+ctx+item) attend to interest.
        query = tokens[:, 0:3, :].mean(dim=1, keepdim=True)  # (B,1,d)
        interest_tok = tokens[:, 4:5, :]
        fused = self.fusion(query, interest_tok).squeeze(1)

        rep = self.out_ln(torch.cat([tokens[:, 2, :], fused], dim=-1))
        logit = self.head(rep).squeeze(-1)
        return logit


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, group_index: torch.Tensor) -> torch.Tensor:
    """Pairwise loss inside one request group.

    labels are assumed in [0,1], with higher meaning positive.
    """
    g_logits = logits[group_index]
    g_labels = labels[group_index]
    pos = g_logits[g_labels > 0.5]
    neg = g_logits[g_labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    # sample pairs
    p = pos.unsqueeze(1)
    n = neg.unsqueeze(0)
    return F.softplus(-(p - n)).mean()


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)
