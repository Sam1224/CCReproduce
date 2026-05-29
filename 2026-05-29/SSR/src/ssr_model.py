from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_sparse(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (topk_indices, topk_values) along last dim.

    This is a convenient sparse representation to avoid constructing huge dense sparse vectors.
    """

    values, indices = torch.topk(x, k=k, dim=-1)
    return indices, values


def sparse_dot(
    idx_a: torch.Tensor,
    val_a: torch.Tensor,
    idx_b: torch.Tensor,
    val_b: torch.Tensor,
) -> torch.Tensor:
    """Compute dot product for two top-k sparse vectors.

    Shapes:
      idx_a, val_a: [..., k]
      idx_b, val_b: [..., k]

    Returns:
      dot: [...]

    Implementation is O(k^2) but k is small in this toy reproduction.
    """

    # [..., k, 1] == [..., 1, k] -> [..., k, k]
    match = idx_a.unsqueeze(-1) == idx_b.unsqueeze(-2)
    prod = val_a.unsqueeze(-1) * val_b.unsqueeze(-2)
    return (prod * match).sum(dim=(-1, -2))


def maxsim_dense(q: torch.Tensor, d: torch.Tensor, q_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    """Dense MaxSim scoring (ColBERT-style).

    q: [B, Lq, H]
    d: [B, Ld, H]
    q_mask: [B, Lq]
    d_mask: [B, Ld]
    """

    sim = torch.einsum("bqh,bdh->bqd", q, d)  # [B, Lq, Ld]

    # mask doc tokens
    sim = sim.masked_fill(~d_mask[:, None, :], float("-inf"))
    best = sim.max(dim=-1).values  # [B, Lq]

    best = best.masked_fill(~q_mask, 0.0)
    return best.sum(dim=-1)  # [B]


def maxsim_sparse(
    q_idx: torch.Tensor,
    q_val: torch.Tensor,
    d_idx: torch.Tensor,
    d_val: torch.Tensor,
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
) -> torch.Tensor:
    """Sparse MaxSim scoring.

    q_idx/q_val: [B, Lq, k]
    d_idx/d_val: [B, Ld, k]
    """

    B, Lq, k = q_idx.shape
    Ld = d_idx.shape[1]

    q_idx_e = q_idx.unsqueeze(2).expand(B, Lq, Ld, k)
    q_val_e = q_val.unsqueeze(2).expand(B, Lq, Ld, k)
    d_idx_e = d_idx.unsqueeze(1).expand(B, Lq, Ld, k)
    d_val_e = d_val.unsqueeze(1).expand(B, Lq, Ld, k)

    dot = sparse_dot(q_idx_e, q_val_e, d_idx_e, d_val_e)  # [B, Lq, Ld]
    dot = dot.masked_fill(~d_mask[:, None, :], float("-inf"))
    best = dot.max(dim=-1).values  # [B, Lq]
    best = best.masked_fill(~q_mask, 0.0)
    return best.sum(dim=-1)


@dataclass
class SSRConfig:
    vocab_size: int
    base_dim: int = 96
    sae_hidden_dim: int = 4096
    topk: int = 32


class SparseAutoEncoder(nn.Module):
    def __init__(self, base_dim: int, hidden_dim: int, topk: int):
        super().__init__()
        self.base_dim = base_dim
        self.hidden_dim = hidden_dim
        self.topk = topk

        self.b_pre = nn.Parameter(torch.zeros(base_dim))
        self.enc = nn.Linear(base_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, base_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_hat, z_idx, z_val).

        x: [..., base_dim]
        x_hat: [..., base_dim]
        z_idx/z_val: [..., topk]
        """

        x_centered = x - self.b_pre
        h = self.enc(x_centered)
        z_idx, z_val = topk_sparse(h, k=self.topk)

        # Build a dense hidden vector for decoding (toy). For real SSR this should stay sparse.
        z_dense = torch.zeros(*h.shape, device=h.device, dtype=h.dtype)
        z_dense.scatter_(dim=-1, index=z_idx, src=z_val)

        x_hat = self.dec(z_dense) + self.b_pre
        return x_hat, z_idx, z_val


class SSRToyModel(nn.Module):
    def __init__(self, cfg: SSRConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.base_dim, padding_idx=0)
        self.sae = SparseAutoEncoder(cfg.base_dim, cfg.sae_hidden_dim, cfg.topk)

    def encode_dense(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(token_ids)

    def encode_sparse(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb(token_ids)
        x_hat, z_idx, z_val = self.sae(x)
        return x_hat, z_idx, z_val

    def loss_recon(self, x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (x - x_hat) ** 2
        diff = diff.sum(dim=-1)
        diff = diff.masked_fill(~mask, 0.0)
        denom = mask.sum().clamp_min(1)
        return diff.sum() / denom

    def contrastive_ce_loss_dense(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.encode_dense(query_ids)
        d = self.encode_dense(doc_ids)

        B = q.shape[0]
        scores = []
        for i in range(B):
            # score query i against all docs in batch
            qi = q[i : i + 1].expand(B, -1, -1)
            qim = query_mask[i : i + 1].expand(B, -1)
            s = maxsim_dense(qi, d, qim, doc_mask)
            scores.append(s)
        logits = torch.stack(scores, dim=0)  # [B, B]
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)

    def contrastive_ce_loss_sparse(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, q_idx, q_val = self.encode_sparse(query_ids)
        _, d_idx, d_val = self.encode_sparse(doc_ids)

        B = q_idx.shape[0]
        scores = []
        for i in range(B):
            qi_idx = q_idx[i : i + 1].expand(B, -1, -1)
            qi_val = q_val[i : i + 1].expand(B, -1, -1)
            qim = query_mask[i : i + 1].expand(B, -1)
            s = maxsim_sparse(qi_idx, qi_val, d_idx, d_val, qim, doc_mask)
            scores.append(s)
        logits = torch.stack(scores, dim=0)
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)
