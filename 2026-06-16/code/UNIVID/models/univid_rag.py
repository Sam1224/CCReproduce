"""
UNIVID-RAG: Retrieval-Augmented Moderation for novel/subtle violations.

Paper (§3.2): "UNIVID-RAG retrieves historically similar violation cases
from a curated case library and uses them as in-context examples for
the moderation decision."

This addresses a core challenge in content moderation: novel violation
patterns not seen during training. By retrieving similar past cases,
UNIVID-RAG can generalize to unseen violation types.

Architecture:
  1. Query encoder: maps current segment → query embedding
  2. Case library: indexed embeddings of historical violation cases
  3. Attention-based aggregator: soft-weighted combination of retrieved cases
  4. Conditioned classifier: moderation head conditioned on retrieved context

Paper result: RAG component contributes +9% recall on novel categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievedCase:
    case_id: str
    category: str
    similarity: float
    embed: torch.Tensor


class CaseLibrary:
    """
    In-memory index of historical violation case embeddings.

    Paper (§3.2): "curated case library updated incrementally with
    expert-reviewed cases."

    In production: backed by FAISS with ~100k entries.
    """

    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim
        self.embeddings: List[torch.Tensor] = []
        self.categories: List[str] = []
        self.case_ids: List[str] = []
        self._index: Optional[torch.Tensor] = None  # (N, D) stacked tensor

    def add(self, case_id: str, category: str, embed: torch.Tensor):
        self.embeddings.append(embed.detach().cpu())
        self.categories.append(category)
        self.case_ids.append(case_id)
        self._index = None  # invalidate cache

    def build(self):
        if self.embeddings:
            self._index = torch.stack(self.embeddings)  # (N, D)
            # L2-normalize for cosine similarity
            self._index = F.normalize(self._index, dim=-1)

    def retrieve(
        self,
        query_embed: torch.Tensor,
        top_k: int = 5,
        device: str = "cpu",
    ) -> List[RetrievedCase]:
        """
        Retrieve top-k most similar historical cases.

        Args:
            query_embed: (D,) L2-normalized query embedding
            top_k: number of cases to retrieve
        """
        if self._index is None or len(self._index) == 0:
            return []

        index = self._index.to(device)
        sims = torch.mv(index, query_embed)  # cosine similarity (index is normalized)
        k = min(top_k, len(sims))
        top_sims, top_idx = sims.topk(k)

        return [
            RetrievedCase(
                case_id=self.case_ids[i],
                category=self.categories[i],
                similarity=top_sims[j].item(),
                embed=self._index[i],
            )
            for j, i in enumerate(top_idx.tolist())
        ]

    def __len__(self):
        return len(self.embeddings)


class QueryEncoder(nn.Module):
    """
    Encodes visual+caption features into a query embedding for retrieval.
    Separate from PolicyCaptionEncoder — optimized for retrieval (InfoNCE).
    """

    def __init__(self, visual_dim: int = 768, caption_dim: int = 512, embed_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim + caption_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, visual_feat: torch.Tensor, caption_embed: torch.Tensor) -> torch.Tensor:
        """Returns (B, embed_dim) L2-normalized embedding."""
        x = torch.cat([visual_feat, caption_embed], dim=-1)
        return F.normalize(self.proj(x), dim=-1)


class RAGAggregator(nn.Module):
    """
    Attention-based aggregation of retrieved cases.

    Paper: soft-attention over top-k retrieved cases,
    query = current segment, keys/values = retrieved case embeddings.

    The aggregated context is concatenated with the query for final classification.
    """

    def __init__(self, embed_dim: int = 512, num_categories: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_categories),
        )
        self.binary_head = nn.Linear(embed_dim * 2, 2)

    def forward(
        self,
        query_embed: torch.Tensor,
        retrieved_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embed:     (B, D)
            retrieved_embeds: (B, K, D) top-k retrieved case embeddings
        Returns:
            binary_logits:   (B, 2)
            category_logits: (B, num_categories)
        """
        q = query_embed.unsqueeze(1)  # (B, 1, D)
        # Cross-attention: query attends to retrieved cases
        attn_out, _ = self.attn(q, retrieved_embeds, retrieved_embeds)  # (B, 1, D)
        context = self.norm(attn_out.squeeze(1))  # (B, D)

        fused = torch.cat([query_embed, context], dim=-1)  # (B, 2D)
        return self.binary_head(fused), self.classifier(fused)


class UniVIDRAG(nn.Module):
    """
    Full UNIVID-RAG module.

    Given a segment (visual + caption), retrieves top-k historical cases
    and produces moderation decisions conditioned on retrieved context.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        caption_dim: int = 512,
        embed_dim: int = 512,
        num_categories: int = 8,
        top_k: int = 5,
    ):
        super().__init__()
        self.query_encoder = QueryEncoder(visual_dim, caption_dim, embed_dim)
        self.aggregator = RAGAggregator(embed_dim, num_categories)
        self.top_k = top_k
        self.embed_dim = embed_dim

    def forward(
        self,
        visual_feat: torch.Tensor,
        caption_embed: torch.Tensor,
        retrieved_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feat:      (B, visual_dim)
            caption_embed:    (B, caption_dim)
            retrieved_embeds: (B, K, embed_dim) — pre-retrieved case embeddings
        Returns:
            query_embed:      (B, embed_dim) for building case library
            binary_logits:    (B, 2)
            category_logits:  (B, num_categories)
        """
        query_embed = self.query_encoder(visual_feat, caption_embed)

        # Pad retrieved_embeds if fewer than top_k cases retrieved
        B, K, D = retrieved_embeds.shape
        if K == 0:
            # No retrieved cases: use zero context
            retrieved_embeds = torch.zeros(B, 1, D, device=visual_feat.device)

        binary_logits, category_logits = self.aggregator(query_embed, retrieved_embeds)
        return query_embed, binary_logits, category_logits


class InfoNCELoss(nn.Module):
    """
    Contrastive loss for training the query encoder.
    Violation cases from the same category should be close;
    safe/different-category cases should be far.

        L_InfoNCE = -log(exp(q·k+/τ) / Σ exp(q·k_i/τ))
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_embeds:    (B, D) queries from current batch
            positive_embeds: (B, D) positive pairs (same category)
        """
        # All-pairs similarity matrix
        logits = torch.mm(query_embeds, positive_embeds.t()) / self.temperature
        labels = torch.arange(len(query_embeds), device=query_embeds.device)
        return F.cross_entropy(logits, labels)
