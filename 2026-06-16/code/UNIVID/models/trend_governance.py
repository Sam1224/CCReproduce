"""
Stage 3: Trend Governance (UNIVID).

Paper (§3.3): "Trend Governance detects emerging violation patterns by
clustering cached UNIVID embeddings over a rolling time window.
When a new cluster surpasses a density threshold, an adaptive trend head
is fine-tuned and deployed to the moderation actor."

This stage operates at a slower cadence (hours/days) vs. the real-time
moderation actor (milliseconds). It is responsible for keeping the system
current as new violation tactics emerge.

Key components:
  1. EmbeddingCache: ring buffer of recent segment embeddings + decisions
  2. TrendDetector: online k-means / DBSCAN to find emerging clusters
  3. AdaptiveTrendHead: lightweight classifier fine-tuned on new cluster
  4. TrendGovernanceSystem: orchestrates the full pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class CachedEmbedding:
    segment_id: str
    embed: torch.Tensor        # (D,)
    decision: str              # "safe" | category string
    timestamp: float
    is_violation: bool


class EmbeddingCache:
    """
    Rolling cache of segment embeddings for trend analysis.

    Paper: embeddings from the moderation actor are cached and
    periodically analyzed for emerging patterns.
    """

    def __init__(self, max_size: int = 50000, embed_dim: int = 512):
        self.max_size = max_size
        self.embed_dim = embed_dim
        self._cache: deque = deque(maxlen=max_size)

    def add(self, entry: CachedEmbedding):
        self._cache.append(entry)

    def get_embeddings(
        self,
        violations_only: bool = False,
        last_n: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Return stacked embeddings and corresponding labels."""
        entries = list(self._cache)
        if violations_only:
            entries = [e for e in entries if e.is_violation]
        if last_n is not None:
            entries = entries[-last_n:]

        if not entries:
            return torch.zeros(0, self.embed_dim), []

        embeds = torch.stack([e.embed for e in entries])
        labels = [e.decision for e in entries]
        return embeds, labels

    def __len__(self):
        return len(self._cache)


class TrendDetector:
    """
    Online clustering to detect emerging violation trends.

    Paper: "We apply DBSCAN over recent embeddings. Dense clusters
    with no matching existing category trigger alert for human review
    and adaptive head fine-tuning."

    Simplified: uses k-means-style centroid comparison.
    Production: DBSCAN with approximate NN index (FAISS).
    """

    def __init__(
        self,
        density_threshold: int = 50,    # min cluster size to be a "trend"
        similarity_threshold: float = 0.85,  # min sim to existing category centroid
        embed_dim: int = 512,
    ):
        self.density_threshold = density_threshold
        self.similarity_threshold = similarity_threshold
        self.embed_dim = embed_dim
        self.known_centroids: Dict[str, torch.Tensor] = {}

    def update_known_centroids(self, embeddings: torch.Tensor, labels: List[str]):
        """Update centroids from labeled data."""
        unique_labels = set(labels)
        for lbl in unique_labels:
            mask = torch.tensor([l == lbl for l in labels])
            centroid = F.normalize(embeddings[mask].mean(0, keepdim=True), dim=-1).squeeze(0)
            self.known_centroids[lbl] = centroid

    def detect_novel_clusters(
        self,
        embeddings: torch.Tensor,
        n_clusters: int = 10,
    ) -> List[Dict]:
        """
        Detect clusters in the embedding space that don't match known categories.

        Returns list of dicts: {centroid, size, max_known_sim, is_novel}
        """
        if len(embeddings) < self.density_threshold:
            return []

        # Simple k-means (production: DBSCAN)
        normed = F.normalize(embeddings, dim=-1)
        centroids = self._kmeans(normed, n_clusters, n_iter=50)

        results = []
        for i, centroid in enumerate(centroids):
            # Assign embeddings to this centroid
            sims = torch.mv(normed, centroid)
            cluster_mask = sims > 0.6
            cluster_size = cluster_mask.sum().item()

            if cluster_size < self.density_threshold:
                continue

            # Check similarity to known categories
            max_known_sim = 0.0
            if self.known_centroids:
                known_stack = torch.stack(list(self.known_centroids.values()))
                known_sims = torch.mv(known_stack, centroid)
                max_known_sim = known_sims.max().item()

            is_novel = max_known_sim < self.similarity_threshold
            results.append({
                "cluster_id": i,
                "centroid": centroid,
                "size": int(cluster_size),
                "max_known_sim": max_known_sim,
                "is_novel": is_novel,
            })

        return results

    @staticmethod
    def _kmeans(
        embeddings: torch.Tensor,
        n_clusters: int,
        n_iter: int = 50,
    ) -> torch.Tensor:
        """Mini k-means; returns (n_clusters, D) centroids."""
        n = len(embeddings)
        if n < n_clusters:
            n_clusters = n
        idx = torch.randperm(n)[:n_clusters]
        centroids = embeddings[idx].clone()

        for _ in range(n_iter):
            # Assign
            sims = torch.mm(embeddings, centroids.t())  # (N, K)
            assignments = sims.argmax(dim=-1)           # (N,)
            # Update
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = F.normalize(embeddings[mask].mean(0), dim=0)
                else:
                    new_centroids[k] = centroids[k]
            centroids = new_centroids

        return centroids  # (K, D)


class AdaptiveTrendHead(nn.Module):
    """
    Lightweight classifier fine-tuned on newly detected trend clusters.

    Paper: "A small adapter head is rapidly fine-tuned (~minutes) and
    deployed alongside UNIVID-Lite to catch the new violation pattern."

    Architecture: 2-layer MLP over existing UNIVID embedding.
    """

    def __init__(self, embed_dim: int = 512, num_trends: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_trends + 1),  # +1 for "not a trend"
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_trends+1) logits."""
        return self.head(embed)

    def fine_tune(
        self,
        embeds: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 20,
        lr: float = 1e-3,
    ) -> float:
        """Rapid fine-tuning on a small set of trend examples."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.train()
        final_loss = 0.0
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = self(embeds)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        self.eval()
        return final_loss


class TrendGovernanceSystem:
    """
    Orchestrates Stage 3: caching → trend detection → adaptive head deployment.

    Paper (§3.3):
      1. Cache UNIVID embeddings + decisions in rolling window
      2. Periodically run TrendDetector over cached embeddings
      3. Novel clusters → alert for human review
      4. Post-review → fine-tune AdaptiveTrendHead on confirmed cases
      5. Deploy adaptive head to moderation pipeline
    """

    def __init__(
        self,
        embed_dim: int = 512,
        cache_size: int = 50000,
        density_threshold: int = 50,
    ):
        self.cache = EmbeddingCache(max_size=cache_size, embed_dim=embed_dim)
        self.detector = TrendDetector(density_threshold=density_threshold)
        self.active_trend_heads: Dict[str, AdaptiveTrendHead] = {}
        self.embed_dim = embed_dim

    def ingest(self, entry: CachedEmbedding):
        self.cache.add(entry)

    def run_trend_analysis(self, violations_only: bool = True) -> List[Dict]:
        """Detect emerging trends from cached embeddings."""
        embeds, labels = self.cache.get_embeddings(violations_only=violations_only)
        if len(embeds) == 0:
            return []

        # Update known centroids from labeled data
        self.detector.update_known_centroids(embeds, labels)

        # Detect novel clusters
        novel_clusters = self.detector.detect_novel_clusters(embeds)
        return [c for c in novel_clusters if c["is_novel"]]

    def deploy_trend_head(
        self,
        trend_name: str,
        trend_embeds: torch.Tensor,
        trend_labels: torch.Tensor,
    ) -> AdaptiveTrendHead:
        """Fine-tune and register a new adaptive head for a detected trend."""
        head = AdaptiveTrendHead(embed_dim=self.embed_dim)
        loss = head.fine_tune(trend_embeds, trend_labels)
        self.active_trend_heads[trend_name] = head
        print(f"Deployed trend head '{trend_name}' | fine-tune loss: {loss:.4f}")
        return head

    def check_trend_heads(
        self,
        embed: torch.Tensor,
        threshold: float = 0.5,
    ) -> Optional[str]:
        """
        Check if embed matches any active trend head.
        Returns trend name if detected, else None.
        """
        for name, head in self.active_trend_heads.items():
            head.eval()
            with torch.no_grad():
                logits = head(embed.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[0]
                if probs[:-1].max().item() > threshold:  # exclude "not a trend" class
                    return name
        return None
