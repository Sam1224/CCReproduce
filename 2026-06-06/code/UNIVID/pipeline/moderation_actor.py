"""
UNIVID Stage (B): Moderation Actor
Two sub-components:
    - UNIVID-Lite: lightweight classifier for high-throughput moderation decisions
    - UNIVID-RAG:  retrieval-augmented model that recalls prior violative events
                   to reduce leakage on rare/novel violations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class UNIVIDLite(nn.Module):
    """
    Lightweight binary classifier per moderation category.
    Takes UNIVID caption embedding + policy context.
    """

    def __init__(self, embed_dim: int = 256, num_categories: int = 64):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(num_categories)
        ])

    def forward(self, caption_embedding: torch.Tensor) -> torch.Tensor:
        """Returns per-category violation probabilities: (B, num_categories)"""
        scores = [head(caption_embedding) for head in self.heads]
        return torch.cat(scores, dim=-1).sigmoid()  # (B, C)


class ViolationMemoryBank:
    """
    In-memory store of past violative video caption embeddings.
    Used by UNIVID-RAG for similarity-based recall.

    In production: backed by a vector database (e.g., Faiss, Milvus).
    """

    def __init__(self, max_size: int = 100_000, embed_dim: int = 256):
        self.max_size = max_size
        self.embed_dim = embed_dim
        self._bank: Optional[torch.Tensor] = None  # (N, embed_dim)
        self._labels: Optional[torch.Tensor] = None  # (N,) violation category

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor):
        if self._bank is None:
            self._bank = embeddings.detach().cpu()
            self._labels = labels.detach().cpu()
        else:
            self._bank = torch.cat([self._bank, embeddings.detach().cpu()], dim=0)
            self._labels = torch.cat([self._labels, labels.detach().cpu()], dim=0)
        if len(self._bank) > self.max_size:
            self._bank = self._bank[-self.max_size:]
            self._labels = self._labels[-self.max_size:]

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (top_k_embeddings, top_k_labels) for each query."""
        if self._bank is None:
            B = query.shape[0]
            return (
                torch.zeros(B, top_k, self.embed_dim),
                torch.zeros(B, top_k, dtype=torch.long),
            )
        sim = F.cosine_similarity(
            query.cpu().unsqueeze(1),          # (B, 1, D)
            self._bank.unsqueeze(0),           # (1, N, D)
            dim=-1
        )  # (B, N)
        top_idx = sim.topk(min(top_k, sim.shape[1]), dim=-1).indices  # (B, k)
        retrieved_emb = self._bank[top_idx]    # (B, k, D)
        retrieved_lbl = self._labels[top_idx]  # (B, k)
        return retrieved_emb, retrieved_lbl


class UNIVID_RAG(nn.Module):
    """
    Retrieval-augmented moderation model.
    Fuses query caption embedding with retrieved violative event embeddings
    to make a recall-enhanced moderation decision.

    Formula (from paper intuition):
        context = Aggregate(retrieved_embeddings)
        decision = Classifier(cat(caption_emb, context))
    """

    def __init__(self, embed_dim: int = 256, top_k: int = 5, num_categories: int = 64):
        super().__init__()
        self.memory_bank = ViolationMemoryBank(embed_dim=embed_dim)
        self.top_k = top_k

        # Context aggregator
        self.context_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Decision head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories),
        )

    def forward(self, caption_embedding: torch.Tensor) -> torch.Tensor:
        retrieved_emb, _ = self.memory_bank.retrieve(caption_embedding, self.top_k)
        retrieved_emb = retrieved_emb.to(caption_embedding.device)

        # Attend from caption to retrieved context
        query = caption_embedding.unsqueeze(1)  # (B, 1, D)
        context, _ = self.context_attn(query, retrieved_emb, retrieved_emb)
        context = context.squeeze(1)  # (B, D)

        fused = torch.cat([caption_embedding, context], dim=-1)  # (B, 2D)
        return self.classifier(fused).sigmoid()  # (B, C)


class ModerationActor(nn.Module):
    """
    Combines UNIVID-Lite and UNIVID-RAG outputs.
    Final decision is the max across both (recall-focused union).
    """

    def __init__(self, embed_dim: int = 256, num_categories: int = 64):
        super().__init__()
        self.lite = UNIVIDLite(embed_dim, num_categories)
        self.rag = UNIVID_RAG(embed_dim, top_k=5, num_categories=num_categories)

    def forward(self, caption_embedding: torch.Tensor) -> dict:
        lite_scores = self.lite(caption_embedding)   # (B, C)
        rag_scores = self.rag(caption_embedding)     # (B, C)

        # Union recall: flag if either model predicts violation (max ensemble)
        combined = torch.maximum(lite_scores, rag_scores)  # (B, C)

        return {
            "lite_scores": lite_scores,
            "rag_scores": rag_scores,
            "combined_scores": combined,
        }
