"""
Violation Case Store — MLLM-Boosted Similarity Matching (Path 2)
From arXiv 2512.03553 §3.3

The violation store maintains a curated set of known policy violations.
For each new piece of live content, we compute its embedding and find
the nearest violation case via kNN search.

In production (paper §3.3):
  - Store contains embeddings of ~100K known violation examples
  - Embeddings are MLLM-enhanced (distilled from MLLM's richer representations)
  - Updated continuously as new violations are confirmed
  - Supports both semantic embeddings (content meaning) and perceptual embeddings (visual/audio features)

The key insight: this path generalizes to NOVEL violations not seen during
training, as long as they are semantically similar to stored examples.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class ViolationCase:
    """A single known violation case in the store."""
    case_id: str
    violation_type: str
    embedding: torch.Tensor          # [embed_dim] — L2-normalized
    mllm_description: str            # MLLM-generated description
    confidence: float = 1.0
    review_count: int = 0


class ViolationStore:
    """
    Curated store of known violation embeddings.
    Supports approximate nearest-neighbor search for Path 2.

    Paper §3.3: "The similarity pipeline leverages a similarity matching engine
    that compares incoming content to a curated set of known policy violations
    using semantic and perceptual embeddings, enabling the system to generalize
    to previously unseen behaviors through nearest-neighbor retrieval."
    """

    def __init__(self, embed_dim: int = 128, use_faiss: bool = False):
        self.embed_dim = embed_dim
        self.use_faiss = use_faiss
        self.cases: List[ViolationCase] = []
        self._embeddings_cache: Optional[torch.Tensor] = None

        # Try to use FAISS for fast ANN search
        if use_faiss:
            try:
                import faiss
                self.index = faiss.IndexFlatIP(embed_dim)  # Inner product (cosine with L2-norm)
                self.faiss = faiss
                print("Using FAISS for ANN search.")
            except ImportError:
                print("FAISS not available — using brute-force cosine similarity.")
                self.use_faiss = False
                self.index = None

    def add_case(self, case: ViolationCase):
        """Add a new violation case to the store."""
        self.cases.append(case)
        self._embeddings_cache = None  # Invalidate cache

        if self.use_faiss and self.index is not None:
            emb = case.embedding.cpu().numpy().reshape(1, -1).astype(np.float32)
            self.index.add(emb)

    def add_batch(self, embeddings: torch.Tensor, violation_types: List[str], descriptions: List[str]):
        """Add multiple violation cases at once."""
        for i in range(embeddings.size(0)):
            case = ViolationCase(
                case_id=f"case_{len(self.cases):06d}",
                violation_type=violation_types[i],
                embedding=embeddings[i].cpu(),
                mllm_description=descriptions[i],
            )
            self.add_case(case)

    def get_embeddings(self) -> Optional[torch.Tensor]:
        """Return all stored embeddings as a tensor [N, D]."""
        if len(self.cases) == 0:
            return None
        if self._embeddings_cache is None:
            self._embeddings_cache = torch.stack([c.embedding for c in self.cases])
        return self._embeddings_cache

    def search(
        self,
        query_embeds: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, List[List[ViolationCase]]]:
        """
        Find top-k nearest violation cases for each query.
        Returns: (scores [B, k], cases [[ViolationCase]*k]*B)
        """
        if len(self.cases) == 0:
            B = query_embeds.size(0)
            return torch.zeros(B, top_k), [[] * top_k] * B

        if self.use_faiss and self.index is not None:
            q = query_embeds.cpu().numpy().astype(np.float32)
            scores, indices = self.index.search(q, min(top_k, len(self.cases)))
            scores = torch.from_numpy(scores)
            top_cases = [[self.cases[idx] for idx in row if idx >= 0] for row in indices]
        else:
            # Brute-force cosine similarity
            store_embeds = self.get_embeddings().to(query_embeds.device)
            sim = torch.matmul(query_embeds, store_embeds.t())  # [B, N]
            k = min(top_k, len(self.cases))
            scores, indices = sim.topk(k, dim=-1)
            top_cases = [[self.cases[idx.item()] for idx in row] for row in indices]

        return scores, top_cases

    def get_decision_stats(
        self,
        query_embeds: torch.Tensor,
        threshold: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Path 2 decision statistics for a batch of queries.
        Returns max similarity score and the dominant violation type.
        """
        scores, top_cases = self.search(query_embeds, top_k=1)
        max_scores = scores[:, 0] if scores.size(1) > 0 else torch.zeros(query_embeds.size(0))
        dominant_types = [
            cases[0].violation_type if cases else "unknown"
            for cases in top_cases
        ]
        is_violation = max_scores > threshold
        return {
            "max_similarity": max_scores,
            "is_violation": is_violation,
            "dominant_type": dominant_types,
        }

    def __len__(self) -> int:
        return len(self.cases)

    def summary(self) -> Dict[str, int]:
        from collections import Counter
        type_counts = Counter(c.violation_type for c in self.cases)
        return dict(type_counts)


def build_toy_violation_store(
    model: torch.nn.Module,
    embed_dim: int = 128,
    num_per_class: int = 10,
) -> ViolationStore:
    """
    Build a toy violation store with randomly sampled embeddings.
    In production: uses MLLM to embed curated violation examples.
    """
    from model.dual_path import VIOLATION_CLASSES

    store = ViolationStore(embed_dim=embed_dim)

    violation_types = [c for c in VIOLATION_CLASSES if c != "compliant"]
    for vtype in violation_types:
        embeddings = torch.randn(num_per_class, embed_dim)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        descriptions = [
            f"[Mock] MLLM description for {vtype} case {i}" for i in range(num_per_class)
        ]
        store.add_batch(embeddings, [vtype] * num_per_class, descriptions)

    print(f"Built violation store: {len(store)} cases")
    print(f"Distribution: {store.summary()}")
    return store
