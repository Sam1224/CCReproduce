"""
UNIVID Stage (C): Trend Governance
Uses cached UNIVID caption embeddings to detect emerging violation trends
via online clustering, without requiring new model training.

Formula:
    - Maintain a rolling buffer of caption embeddings
    - Run k-means / DBSCAN to identify dense clusters
    - Clusters that grow rapidly and are not covered by existing categories
      are flagged as emerging trends
    - A lightweight trend head is fine-tuned on flagged cluster members
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class EmbeddingBuffer:
    """Rolling buffer of (caption_embedding, timestamp, category) tuples."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._embeddings: deque = deque(maxlen=capacity)
        self._timestamps: deque = deque(maxlen=capacity)
        self._categories: deque = deque(maxlen=capacity)

    def add(
        self,
        embedding: np.ndarray,   # (D,)
        timestamp: float,
        category: int = -1,      # -1 = unknown / potentially novel
    ):
        self._embeddings.append(embedding)
        self._timestamps.append(timestamp)
        self._categories.append(category)

    @property
    def embeddings(self) -> np.ndarray:
        return np.stack(list(self._embeddings))  # (N, D)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array(list(self._timestamps))

    def __len__(self) -> int:
        return len(self._embeddings)


class TrendDetector:
    """
    Sliding-window trend detection over UNIVID caption embeddings.

    Method (from paper):
        1. Buffer recent embeddings (last W hours/days)
        2. Cluster with mini-batch k-means
        3. Compare cluster sizes to historical baseline
        4. Flag clusters with growth rate > threshold as emerging trends
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_clusters: int = 32,
        growth_threshold: float = 0.3,
        window_size: int = 10_000,
    ):
        self.embed_dim = embed_dim
        self.n_clusters = n_clusters
        self.growth_threshold = growth_threshold
        self.window_size = window_size
        self.buffer = EmbeddingBuffer(capacity=window_size * 2)
        self._baseline_cluster_sizes: Optional[np.ndarray] = None

    def update(self, embedding: np.ndarray, timestamp: float, category: int = -1):
        self.buffer.add(embedding, timestamp, category)

    def detect_trends(self) -> List[int]:
        """Returns indices of emerging trend clusters."""
        if len(self.buffer) < 1000:
            return []

        embeddings = self.buffer.embeddings

        # Simple k-means via torch (paper uses a more sophisticated online variant)
        emb_t = torch.tensor(embeddings, dtype=torch.float32)
        emb_norm = nn.functional.normalize(emb_t, dim=-1)

        # Initialize centroids with k-means++
        centroids = self._kmeans_pp_init(emb_norm)
        for _ in range(10):  # EM iterations
            dists = torch.cdist(emb_norm, centroids)  # (N, K)
            assignments = dists.argmin(dim=-1)        # (N,)
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = emb_norm[mask].mean(dim=0)

        cluster_sizes = torch.bincount(assignments, minlength=self.n_clusters).numpy()

        if self._baseline_cluster_sizes is None:
            self._baseline_cluster_sizes = cluster_sizes.copy()
            return []

        # Detect growth
        with np.errstate(divide='ignore', invalid='ignore'):
            growth = np.where(
                self._baseline_cluster_sizes > 0,
                (cluster_sizes - self._baseline_cluster_sizes) / self._baseline_cluster_sizes,
                0.0
            )

        emerging = np.where(growth > self.growth_threshold)[0].tolist()
        self._baseline_cluster_sizes = cluster_sizes.copy()
        return emerging

    def _kmeans_pp_init(self, data: torch.Tensor) -> torch.Tensor:
        K = self.n_clusters
        n = data.shape[0]
        idx = torch.randint(n, (1,)).item()
        centroids = [data[idx]]
        for _ in range(1, K):
            dists = torch.stack([
                torch.cdist(data, c.unsqueeze(0)).squeeze(-1)
                for c in centroids
            ], dim=1).min(dim=1).values  # (N,)
            probs = (dists ** 2)
            probs = probs / probs.sum()
            chosen = torch.multinomial(probs, 1).item()
            centroids.append(data[chosen])
        return torch.stack(centroids)  # (K, D)


class TrendHead(nn.Module):
    """
    Lightweight binary classifier fine-tuned on emerging trend clusters.
    One head per detected emerging trend category.
    """

    def __init__(self, embed_dim: int = 256, num_trend_heads: int = 8):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_trend_heads)
        ])

    def forward(self, caption_embedding: torch.Tensor) -> torch.Tensor:
        # (B, num_trend_heads)
        return torch.cat([h(caption_embedding) for h in self.heads], dim=-1)


class TrendGovernance(nn.Module):
    """Full Trend Governance module."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.detector = TrendDetector(embed_dim=embed_dim)
        self.trend_head = TrendHead(embed_dim=embed_dim)

    def update_buffer(self, embedding: torch.Tensor, timestamp: float):
        self.detector.update(embedding.detach().cpu().numpy()[0], timestamp)

    def detect(self) -> List[int]:
        return self.detector.detect_trends()

    def classify_trends(self, caption_embedding: torch.Tensor) -> torch.Tensor:
        return self.trend_head(caption_embedding)  # (B, num_trend_heads)
