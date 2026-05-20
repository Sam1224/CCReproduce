"""Fine-grained clustering: distinguish variants vs. new sub-issues.

Paper §3.2 (Stage 2): Within each coarse cluster, apply fine-grained clustering
to separate:
  - Variants: different presentations of the same underlying issue
  - New sub-issues: genuinely distinct problematic content patterns

The key insight: variants have high intra-cluster semantic similarity but
low diversity of surface forms, while new sub-issues show distinct semantic patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from clustering.coarse_cluster import CoarseCluster


@dataclass
class FineCluster:
    cluster_id: str  # format: "coarse_{coarse_id}_fine_{fine_id}"
    video_ids: list[str]
    descriptions: list[str]
    is_new_issue: bool  # True = new issue, False = variant of existing/coarse issue
    centroid: np.ndarray
    intra_similarity: float  # Average pairwise cosine similarity


_VARIANT_SIMILARITY_THRESHOLD = 0.75  # High similarity = likely variants


def fine_cluster(
    coarse_cluster: CoarseCluster,
    embeddings_map: dict[str, np.ndarray],
    variant_threshold: float = _VARIANT_SIMILARITY_THRESHOLD,
) -> list[FineCluster]:
    """Stage 2: Fine-grained analysis within a coarse cluster.

    For each coarse cluster:
    1. Compute pairwise cosine similarity among member videos
    2. Sub-clusters with high internal similarity → variants
    3. Sub-clusters with distinct patterns → potential new issues
    """
    if len(coarse_cluster.video_ids) < 2:
        # Single-video cluster — treat as potential new issue
        vid_id = coarse_cluster.video_ids[0]
        emb = embeddings_map.get(vid_id, np.zeros(64))
        return [FineCluster(
            cluster_id=f"coarse_{coarse_cluster.cluster_id}_fine_0",
            video_ids=coarse_cluster.video_ids,
            descriptions=coarse_cluster.descriptions,
            is_new_issue=True,
            centroid=emb,
            intra_similarity=1.0,
        )]

    # Build embedding matrix for this cluster
    vids = coarse_cluster.video_ids
    embs = np.stack([embeddings_map.get(v, np.zeros(64)) for v in vids])

    # Pairwise cosine similarity
    sim_matrix = cosine_similarity(embs)
    # Average pairwise sim (excluding diagonal)
    n = len(vids)
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0

    # Simple decision: high avg similarity → variants, low → potential new issues
    # Paper uses more sophisticated LLM-based discrimination
    is_new = avg_sim < variant_threshold

    centroid = embs.mean(axis=0)

    return [FineCluster(
        cluster_id=f"coarse_{coarse_cluster.cluster_id}_fine_0",
        video_ids=vids,
        descriptions=coarse_cluster.descriptions,
        is_new_issue=is_new,
        centroid=centroid,
        intra_similarity=float(avg_sim),
    )]


def classify_clusters(
    coarse_clusters: list[CoarseCluster],
    embeddings_map: dict[str, np.ndarray],
    variant_threshold: float = _VARIANT_SIMILARITY_THRESHOLD,
) -> list[FineCluster]:
    """Apply fine clustering to all coarse clusters."""
    fine_clusters: list[FineCluster] = []
    for cc in coarse_clusters:
        fine_clusters.extend(
            fine_cluster(cc, embeddings_map, variant_threshold)
        )
    return fine_clusters
