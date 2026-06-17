"""
Reference-Based Similarity Matching Pipeline (Pipeline B).

Paper (§3.2): "reference-based similarity matching for novel or subtle cases."
Uses MLLM-distilled embeddings to compare query segments against a bank
of known violation references.

Key concepts:
  - Reference bank: curated library of known violation examples
  - Similarity score: cosine similarity between query and reference embeddings
  - Threshold-based decision: segment flagged if max similarity > threshold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class SimilarityStudent(nn.Module):
    """
    Lightweight student embedding model for similarity matching.

    Trained to produce embeddings aligned with MLLM teacher embeddings
    via contrastive distillation.

    Paper: "MLLM distilling knowledge into [similarity pipeline]"
    """

    def __init__(self, audio_dim=512, visual_dim=768, text_dim=128,
                 hidden_dim=256, embed_dim=512):
        super().__init__()
        input_dim = audio_dim + visual_dim + text_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, text_feat, audio_feat, visual_feat):
        """
        Returns L2-normalized embedding for similarity computation.
        """
        x = torch.cat([text_feat, audio_feat, visual_feat], dim=-1)
        embed = self.encoder(x)
        return F.normalize(embed, dim=-1)  # (B, embed_dim)


class SimilarityDistillationLoss(nn.Module):
    """
    Distillation loss for similarity pipeline.

    Aligns student embeddings with teacher embeddings via MSE + InfoNCE.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_embeds, teacher_embeds):
        """
        Args:
            student_embeds: (B, D) normalized
            teacher_embeds: (B, D) normalized
        """
        # MSE alignment loss
        mse_loss = F.mse_loss(student_embeds, teacher_embeds)

        # InfoNCE: student embed should match its own teacher embed
        logits = torch.mm(student_embeds, teacher_embeds.t()) / self.temperature
        labels = torch.arange(len(student_embeds), device=student_embeds.device)
        nce_loss = F.cross_entropy(logits, labels)

        return mse_loss + nce_loss


class ReferenceBank:
    """
    In-memory reference bank of known violation embeddings.

    Paper (§3.2): "reference-based similarity matching"
    Each entry: {category: [embedding_vectors]}
    """

    def __init__(self, embed_dim=512):
        self.embed_dim = embed_dim
        self.bank: Dict[str, torch.Tensor] = {}  # category -> (N, D) embeddings

    def add(self, category: str, embedding: torch.Tensor):
        """Add a reference embedding to the bank."""
        if category not in self.bank:
            self.bank[category] = []
        self.bank[category].append(embedding.detach().cpu())

    def build(self):
        """Stack per-category embeddings into tensors."""
        for cat in self.bank:
            if isinstance(self.bank[cat], list):
                self.bank[cat] = torch.stack(self.bank[cat])  # (N, D)

    def get_all_refs(self) -> torch.Tensor:
        """Return all reference embeddings as a single tensor (N_total, D)."""
        all_embeds = []
        for cat, embeds in self.bank.items():
            all_embeds.append(embeds)
        if not all_embeds:
            return torch.zeros(0, self.embed_dim)
        return torch.cat(all_embeds, dim=0)

    def query(self, query_embed: torch.Tensor, top_k: int = 5):
        """
        Find top-k most similar references.

        Args:
            query_embed: (D,) normalized query embedding
            top_k: number of top matches to return
        Returns:
            similarities: (top_k,) cosine similarities
            categories: (top_k,) matched category strings
        """
        all_refs = self.get_all_refs().to(query_embed.device)
        if len(all_refs) == 0:
            return torch.zeros(0), []

        # Cosine similarity (both normalized)
        sims = torch.mv(all_refs, query_embed)

        k = min(top_k, len(sims))
        top_sims, top_idx = sims.topk(k)

        # Recover categories for top matches
        all_cats = []
        for cat, embeds in self.bank.items():
            all_cats.extend([cat] * len(embeds))

        top_cats = [all_cats[i] for i in top_idx.tolist()]
        return top_sims, top_cats


class SimilarityDecider:
    """
    Decision logic for the similarity pipeline.

    Paper: similarity score above threshold triggers violation flag.
    Threshold calibrated to achieve target precision (80% in paper).
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def decide(self, similarities: torch.Tensor, categories: List[str]):
        """
        Args:
            similarities: top-k similarity scores
            categories: corresponding category strings
        Returns:
            is_violation: bool
            predicted_category: str or 'safe'
            confidence: float
        """
        if len(similarities) == 0:
            return False, "safe", 0.0

        max_sim = similarities[0].item()
        best_cat = categories[0] if categories else "safe"

        is_violation = max_sim > self.threshold
        return is_violation, best_cat if is_violation else "safe", max_sim
