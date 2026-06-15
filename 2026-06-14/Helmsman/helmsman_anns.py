"""
Helmsman – Toy reproduction illustrating core ideas from:
  "The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale"
  ECNU / SJTU / Xiaohongshu, arXiv 2606.13145

Concepts illustrated:
  1. Clustering-based ANNS index (IVF-style: inverted file with flat cluster centroids)
  2. Adaptive learned pruning (adjusts nprobe based on query difficulty estimate)
  3. Userspace I/O simulation (batched cluster reads instead of random graph traversal)

Reference:
  Helmsman argues that with userspace I/O + learned pruning + GPU-aided index build,
  clustering-based ANNS can satisfy strict online SLAs while cutting DRAM cost by >90%.
"""

from __future__ import annotations

import random
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DIM = 64          # embedding dimension (toy; paper uses 128–256)
NUM_ITEMS = 5000  # corpus size
NUM_CLUSTERS = 50 # number of IVF clusters


# ---------------------------------------------------------------------------
# 1. K-Means clustering (online index build simulation)
# ---------------------------------------------------------------------------

def kmeans(data: np.ndarray, k: int, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Toy K-Means.  In Helmsman the index is rebuilt with GPU-accelerated faiss/cuML."""
    n = data.shape[0]
    # Random initialisation
    centroids = data[np.random.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1)  # (n, k)
        new_labels = dists.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(k):
            members = data[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return centroids, labels


@dataclass
class ClusteringIndex:
    dim: int
    num_clusters: int
    centroids: Optional[np.ndarray] = None           # (num_clusters, dim)
    inverted_lists: list[list[int]] = field(default_factory=list)  # cluster_id → [item_ids]
    vectors: Optional[np.ndarray] = None             # (n, dim)

    def build(self, vectors: np.ndarray) -> None:
        print(f"[Index] Building clustering index: {vectors.shape[0]} vectors, {self.num_clusters} clusters …")
        t0 = time.time()
        self.vectors = vectors
        centroids, labels = kmeans(vectors, self.num_clusters)
        self.centroids = centroids
        self.inverted_lists = [[] for _ in range(self.num_clusters)]
        for item_id, cluster_id in enumerate(labels):
            self.inverted_lists[cluster_id].append(item_id)
        print(f"[Index] Built in {time.time()-t0:.2f}s. Avg cluster size: "
              f"{vectors.shape[0]/self.num_clusters:.1f}")

    def search(self, query: np.ndarray, top_k: int, nprobe: int) -> list[int]:
        """
        Scan `nprobe` nearest clusters (simulated batch I/O) and return top-k item IDs.
        In Helmsman, each cluster is stored on SSD and read as a contiguous block via
        userspace I/O (bypassing the kernel stack) to maximise NVMe bandwidth.
        """
        assert self.centroids is not None, "Index not built yet."
        # Find nearest clusters
        centroid_dists = np.linalg.norm(self.centroids - query, axis=1)
        probe_clusters = centroid_dists.argsort()[:nprobe]

        # Simulated batch cluster read (userspace I/O in the real system)
        candidates: list[tuple[float, int]] = []
        for c in probe_clusters:
            for item_id in self.inverted_lists[c]:
                d = float(np.linalg.norm(self.vectors[item_id] - query))
                candidates.append((d, item_id))

        candidates.sort(key=lambda x: x[0])
        return [item_id for _, item_id in candidates[:top_k]]


# ---------------------------------------------------------------------------
# 2. Learned Pruning – adjust nprobe based on estimated query difficulty
# ---------------------------------------------------------------------------

class AdaptivePruner:
    """
    In Helmsman, a small model learns to predict how many clusters to probe
    based on the query distribution and desired recall target.
    Here we implement a heuristic proxy: queries with higher entropy (i.e.,
    the top-1 centroid distance is not much smaller than top-2) are deemed
    harder and get a higher nprobe.

    Reference: "Leveling-Learned Pruning" adapts nprobe to query × top-k.
    """

    def __init__(
        self,
        base_nprobe: int = 5,
        max_nprobe: int = 20,
        difficulty_threshold: float = 0.15,
    ) -> None:
        self.base_nprobe = base_nprobe
        self.max_nprobe = max_nprobe
        self.difficulty_threshold = difficulty_threshold

    def predict_nprobe(self, query: np.ndarray, centroids: np.ndarray) -> int:
        dists = np.sort(np.linalg.norm(centroids - query, axis=1))
        if len(dists) < 2:
            return self.base_nprobe
        # Ratio of top-1 to top-2 distance — low ratio means hard query
        difficulty = 1.0 - (dists[0] / (dists[1] + 1e-9))
        if difficulty > self.difficulty_threshold:
            # Hard query → probe more clusters
            nprobe = min(self.max_nprobe, self.base_nprobe * 2)
        else:
            nprobe = self.base_nprobe
        return nprobe


# ---------------------------------------------------------------------------
# 3. Benchmark: compare fixed nprobe vs adaptive
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    return len(set(retrieved[:k]) & set(ground_truth)) / min(k, len(ground_truth))


def run_benchmark(index: ClusteringIndex, queries: np.ndarray, top_k: int = 10) -> None:
    pruner = AdaptivePruner(base_nprobe=5, max_nprobe=20)

    # Ground truth by brute-force
    def brute_force(q: np.ndarray) -> list[int]:
        dists = np.linalg.norm(index.vectors - q, axis=1)
        return dists.argsort()[:top_k].tolist()

    fixed_recalls, adaptive_recalls = [], []
    fixed_nprobes, adaptive_nprobes = [], []

    for q in queries:
        gt = brute_force(q)

        nprobe_fixed = 5
        nprobe_adaptive = pruner.predict_nprobe(q, index.centroids)

        r_fixed = index.search(q, top_k, nprobe_fixed)
        r_adaptive = index.search(q, top_k, nprobe_adaptive)

        fixed_recalls.append(recall_at_k(r_fixed, gt, top_k))
        adaptive_recalls.append(recall_at_k(r_adaptive, gt, top_k))
        fixed_nprobes.append(nprobe_fixed)
        adaptive_nprobes.append(nprobe_adaptive)

    print(f"\n=== Recall@{top_k} Comparison ===")
    print(f"  Fixed nprobe={5}:   recall = {sum(fixed_recalls)/len(fixed_recalls):.3f}  "
          f"avg_nprobe = {sum(fixed_nprobes)/len(fixed_nprobes):.1f}")
    print(f"  Adaptive pruning:  recall = {sum(adaptive_recalls)/len(adaptive_recalls):.3f}  "
          f"avg_nprobe = {sum(adaptive_nprobes)/len(adaptive_nprobes):.1f}")
    print("  (adaptive trades modest extra probing on hard queries for higher recall)")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    print(f"Generating {NUM_ITEMS} random {DIM}-d embeddings …")
    corpus = np.random.randn(NUM_ITEMS, DIM).astype(np.float32)

    # Build index
    index = ClusteringIndex(dim=DIM, num_clusters=NUM_CLUSTERS)
    index.build(corpus)

    # Generate test queries
    NUM_QUERIES = 100
    queries = np.random.randn(NUM_QUERIES, DIM).astype(np.float32)

    run_benchmark(index, queries, top_k=10)

    # --- Simulate GPU-accelerated incremental rebuild ---
    print("\n=== Simulating incremental index rebuild (new embedding model) ===")
    new_corpus = corpus + np.random.randn(*corpus.shape).astype(np.float32) * 0.1
    t0 = time.time()
    index.build(new_corpus)  # In production, Helmsman uses GPU to rebuild in hours
    print(f"Rebuild time (toy, CPU): {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
