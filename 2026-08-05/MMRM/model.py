from __future__ import annotations

"""MMRM: Multiplex Multimodal Representation Model (toy).

This is a *minimal runnable* PyTorch implementation that preserves the paper's core shape:

1) Shared multimodal backbone for items (text + image) and shared query encoder.
2) Task tokens for multi-task ranking (e.g., click / purchase).
3) Multiplex item representations: each item is expanded into K representations.
4) Multiplex user representations: from behavior sequence, pooled per multiplex channel.
5) Multi-task ranking: per-task scores are a weighted mixture of multiplex channel matches.

What is NOT implemented (and why):
- Production-scale multimodal backbones (ViT/BERT/etc.). Here we use embedding mean-pool + MLP.
- Complex multiplex routing / transformer blocks described in many industrial variants.
  Here multiplex is implemented as: `item_shared + mux_token[k] -> MLP`.

Pseudo-code reference (closer to the paper's conceptual view):

    x_item_shared = SharedBackbone(item_text, item_image)
    item_rep[k] = MultiplexBlock(x_item_shared, mux_token[k])

    user_rep[k] = AttentionPool(query, history_item_rep[:, :, k, :])

    alpha_t = softmax(sim(task_token[t], mux_token[:]))
    score_t = sum_k alpha_t[k] * dot(user_rep[k] + query, item_rep[k])

All tensors in this toy are dense; indexing from catalog tensors simulates feature lookup.
"""

from typing import Dict, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class SharedMultimodalBackbone(nn.Module):
    """Shared backbone for text+image item encoding."""

    def __init__(self, vocab_size: int, image_dim: int, hidden_dim: int):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fuse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token ids by mean pooling.

        Args:
            token_ids: (..., seq_len)
        Returns:
            text_emb: (..., hidden_dim)
        """

        emb = self.text_embedding(token_ids)  # (..., seq_len, D)
        pooled = emb.mean(dim=-2)
        return self.text_proj(pooled)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image feature vector.

        Args:
            image: (..., image_dim)
        Returns:
            img_emb: (..., hidden_dim)
        """

        return self.image_proj(image)

    def forward(self, item_text_ids: torch.Tensor, item_image: torch.Tensor) -> torch.Tensor:
        text = self.encode_text(item_text_ids)
        image = self.encode_image(item_image)
        fused = text + image
        return self.fuse(fused)


class MultiplexItemEncoder(nn.Module):
    """Generate multiplex item representations.

    Given shared item embedding x (.., D), output (.., K, D).

    Toy implementation:
        item_rep[k] = MLP(x + mux_token[k])

    In paper/industrial implementations, this could be replaced by cross-attention or routing.
    """

    def __init__(self, hidden_dim: int, num_multiplex: int):
        super().__init__()
        self.num_multiplex = num_multiplex
        self.mux_tokens = nn.Parameter(torch.randn(num_multiplex, hidden_dim) * (hidden_dim**-0.5))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, shared_item: torch.Tensor) -> torch.Tensor:
        # shared_item: (..., D)
        x = shared_item.unsqueeze(-2) + self.mux_tokens  # (..., K, D)
        x = self.mlp(x)
        return _l2norm(x, dim=-1)


class MultiplexUserEncoder(nn.Module):
    """Build multiplex user representations from behavior sequence.

    Inputs:
        query: (B, D)
        history_multi: (B, L, K, D)

    Outputs:
        user_multi: (B, K, D)

    Toy implementation uses per-channel attention pooling over the history length L.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, history_multi: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, K, D = history_multi.shape
        q = _l2norm(self.query_proj(query), dim=-1)  # (B, D)

        # (B, L, K)
        scores = (history_multi * q[:, None, None, :]).sum(dim=-1) / (D**0.5)
        if history_mask is not None:
            scores = scores.masked_fill(~history_mask[:, :, None], -1e9)
        weights = torch.softmax(scores, dim=1)  # along L

        user_multi = (weights[..., None] * history_multi).sum(dim=1)  # (B, K, D)
        return _l2norm(user_multi, dim=-1)


class MMRM(nn.Module):
    """Toy MMRM model for multi-task e-commerce ranking."""

    def __init__(
        self,
        *,
        vocab_size: int,
        image_dim: int,
        hidden_dim: int = 64,
        num_multiplex: int = 4,
        tasks: Tuple[str, ...] = ("click", "purchase"),
    ):
        super().__init__()
        self.tasks = tasks
        self.backbone = SharedMultimodalBackbone(vocab_size=vocab_size, image_dim=image_dim, hidden_dim=hidden_dim)
        self.item_mux = MultiplexItemEncoder(hidden_dim=hidden_dim, num_multiplex=num_multiplex)
        self.user_mux = MultiplexUserEncoder(hidden_dim=hidden_dim)

        # task tokens: (T, D)
        self.task_tokens = nn.Parameter(torch.randn(len(tasks), hidden_dim) * (hidden_dim**-0.5))

        # (optional) a small projection for matching
        self.match_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def task_mixing_weights(self) -> torch.Tensor:
        """Compute per-task mixture weights over multiplex channels.

        Returns:
            alpha: (T, K) with softmax over K.

        We use similarity between task_tokens and mux_tokens:
            alpha[t, k] = softmax_k( <task_token[t], mux_token[k]> )
        """

        # (T, K)
        sim = self.task_tokens @ self.item_mux.mux_tokens.t()
        return torch.softmax(sim, dim=-1)

    def encode_query(self, query_text_ids: torch.Tensor) -> torch.Tensor:
        return _l2norm(self.backbone.encode_text(query_text_ids), dim=-1)

    def _encode_items_from_catalog(
        self,
        item_ids: torch.Tensor,
        catalog_text_ids: torch.Tensor,
        catalog_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup catalog features by ids."""

        text = catalog_text_ids[item_ids]
        image = catalog_image[item_ids]
        return text, image

    def rank_logits(
        self,
        *,
        query_text_ids: torch.Tensor,  # (B, QL)
        history_item_ids: torch.Tensor,  # (B, L)
        candidate_item_ids: torch.Tensor,  # (B, C)
        catalog_text_ids: torch.Tensor,  # (N, TL)
        catalog_image: torch.Tensor,  # (N, Img)
    ) -> Dict[str, torch.Tensor]:
        """Compute per-task ranking logits.

        Returns:
            logits: dict(task -> (B, C))
        """

        B, L = history_item_ids.shape
        _, C = candidate_item_ids.shape

        query = self.encode_query(query_text_ids)  # (B, D)

        # ----- encode history items with shared backbone -----
        hist_text, hist_image = self._encode_items_from_catalog(history_item_ids, catalog_text_ids, catalog_image)
        hist_shared = self.backbone(hist_text, hist_image)  # (B, L, D)
        hist_multi = self.item_mux(hist_shared)  # (B, L, K, D)

        # ----- encode candidate items with shared backbone -----
        cand_text, cand_image = self._encode_items_from_catalog(candidate_item_ids, catalog_text_ids, catalog_image)
        cand_shared = self.backbone(cand_text, cand_image)  # (B, C, D)
        cand_multi = self.item_mux(cand_shared)  # (B, C, K, D)

        # ----- multiplex user representation from behavior sequence -----
        user_multi = self.user_mux(query, hist_multi)  # (B, K, D)

        # ----- per-task scoring -----
        alpha = self.task_mixing_weights()  # (T, K)
        q_plus_u = _l2norm(user_multi + query[:, None, :], dim=-1)  # (B, K, D)
        q_plus_u = self.match_proj(q_plus_u)

        # match: (B, C, K)
        match = (q_plus_u[:, None, :, :] * cand_multi).sum(dim=-1)

        logits: Dict[str, torch.Tensor] = {}
        for t, task_name in enumerate(self.tasks):
            logits[task_name] = (match * alpha[t][None, None, :]).sum(dim=-1)
        return logits

    def forward(
        self,
        query_text_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        catalog_text_ids: torch.Tensor,
        catalog_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.rank_logits(
            query_text_ids=query_text_ids,
            history_item_ids=history_item_ids,
            candidate_item_ids=candidate_item_ids,
            catalog_text_ids=catalog_text_ids,
            catalog_image=catalog_image,
        )


@torch.no_grad()
def ndcg_at_k(scores: torch.Tensor, relevance: torch.Tensor, k: int = 10) -> float:
    """Compute mean NDCG@k for a batch.

    Args:
        scores: (B, C)
        relevance: (B, C) non-negative relevance (float or int)

    Returns:
        scalar float
    """

    B, C = scores.shape
    k = min(k, C)

    order = torch.argsort(scores, dim=-1, descending=True)
    rel_sorted = torch.gather(relevance, dim=-1, index=order)[:, :k]

    denom = torch.log2(torch.arange(k, device=scores.device, dtype=torch.float32) + 2.0)  # (k,)
    dcg = ((2.0**rel_sorted - 1.0) / denom[None, :]).sum(dim=-1)  # (B,)

    ideal_order = torch.argsort(relevance, dim=-1, descending=True)
    ideal_rel = torch.gather(relevance, dim=-1, index=ideal_order)[:, :k]
    idcg = ((2.0**ideal_rel - 1.0) / denom[None, :]).sum(dim=-1)

    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    return ndcg.mean().item()


@torch.no_grad()
def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUC for binary labels.

    This implementation uses the rank-statistics formula (Mann-Whitney U).
    It ignores tie-handling details (ties are rare for continuous scores).

    Args:
        scores: (N,)
        labels: (N,) in {0,1}
    """

    scores = scores.flatten()
    labels = labels.flatten().to(torch.float32)

    pos = labels > 0.5
    n_pos = int(pos.sum().item())
    n = labels.numel()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # ranks: 1..N
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32)

    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())
