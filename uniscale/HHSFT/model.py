from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class HeteroFeatureEncoder(nn.Module):
    """Sparse+Dense encoder (toy)."""

    def __init__(self, sparse_vocab: int, num_fields: int, d_model: int, dense_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(sparse_vocab, d_model)
        self.dense_proj = nn.Linear(dense_dim, d_model)
        self.field_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        # sparse: (B,F), dense: (B,D)
        x = self.emb(sparse)  # (B,F,d)
        x, _ = self.field_attn(x, x, x)
        x = x.mean(dim=1) + self.dense_proj(dense)
        return self.out(torch.tanh(x))


class EntireSpaceInterestFusion(nn.Module):
    """History -> next-interest embedding (toy)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B,L,d)
        lengths = mask.sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, last = self.gru(packed)
        last = last.squeeze(0)  # (B,d)
        pooled = (h * mask.unsqueeze(-1).type_as(h)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        g = self.gate(torch.cat([last, pooled], dim=-1))
        return g * last + (1 - g) * pooled


@dataclass
class RankingOutput:
    pos_score: torch.Tensor  # (B,)
    neg_score: torch.Tensor  # (B,)


class OneModel(nn.Module):
    """UniScale HHSFT (toy).

    Preserves the paper's core structure:
    - Heterogeneous hierarchical feature fusion (field attention + dense projection)
    - Entire-space user interest fusion (sequence model + gating)
    - Ranking head
    """

    def __init__(
        self,
        sparse_vocab: int = 1000,
        num_fields: int = 8,
        dense_dim: int = 16,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.item_encoder = HeteroFeatureEncoder(sparse_vocab, num_fields, d_model, dense_dim)
        self.history_item_encoder = HeteroFeatureEncoder(sparse_vocab, num_fields, d_model, dense_dim)
        self.interest_fusion = EntireSpaceInterestFusion(d_model)

        self.rank_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _score(self, user_interest: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        return self.rank_head(torch.cat([user_interest, item_vec], dim=-1)).squeeze(-1)

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> RankingOutput:
        history_sparse = batch["history_sparse"]  # (B,L,F)
        history_dense = batch["history_dense"]  # (B,L,D)
        history_mask = batch["history_mask"]  # (B,L)

        # Encode each history item.
        bsz, l, f = history_sparse.shape
        hist_flat_sparse = history_sparse.reshape(bsz * l, f)
        hist_flat_dense = history_dense.reshape(bsz * l, history_dense.shape[-1])
        hist_item_vec = self.history_item_encoder(hist_flat_sparse, hist_flat_dense).reshape(bsz, l, -1)

        user_interest = self.interest_fusion(hist_item_vec, history_mask)

        pos_item = self.item_encoder(batch["pos_sparse"], batch["pos_dense"])
        neg_item = self.item_encoder(batch["neg_sparse"], batch["neg_dense"])

        pos_score = self._score(user_interest, pos_item)
        neg_score = self._score(user_interest, neg_item)

        return RankingOutput(pos_score=pos_score, neg_score=neg_score)


def ranking_loss(out: RankingOutput, label: torch.Tensor, margin: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Pointwise (pos better than neg): y in {0,1}
    pos_prob = torch.sigmoid(out.pos_score)
    point = F.binary_cross_entropy(pos_prob, label)

    # Pairwise hinge
    pair = F.relu(margin - (out.pos_score - out.neg_score)).mean()

    total = point + pair
    return total, {"point_loss": float(point.detach()), "pair_loss": float(pair.detach())}
