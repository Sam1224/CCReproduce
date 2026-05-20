"""Coarse-grained clustering: group recalled videos by broad issue type.

Paper §3.2 (Stage 1): Apply coarse clustering to group videos with similar
problematic content patterns before fine-grained analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from sklearn.cluster import KMeans


@dataclass
class CoarseCluster:
    cluster_id: int
    video_ids: list[str]
    descriptions: list[str]
    centroid: np.ndarray


def coarse_cluster(
    video_ids: list[str],
    descriptions: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    n_clusters: int = 10,
    method: str = "kmeans",
) -> list[CoarseCluster]:
    """Stage 1 clustering: group videos by broad issue type.

    Paper uses HDBSCAN for density-based clustering. We fall back to KMeans
    when hdbscan is not installed.

    Args:
        video_ids: List of video identifiers.
        descriptions: Video text descriptions.
        embeddings: L2-normalized embedding matrix (N x D).
        min_cluster_size: Minimum cluster size for HDBSCAN.
        n_clusters: Number of clusters for KMeans fallback.
        method: 'hdbscan' or 'kmeans'.

    Returns:
        List of CoarseCluster objects (noise points excluded).
    """
    if method == "hdbscan" and HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
    else:
        n_clust = min(n_clusters, len(embeddings) // 2)
        kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

    clusters: dict[int, CoarseCluster] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # HDBSCAN noise
            continue
        if label not in clusters:
            clusters[label] = CoarseCluster(
                cluster_id=label,
                video_ids=[],
                descriptions=[],
                centroid=np.zeros(embeddings.shape[1]),
            )
        clusters[label].video_ids.append(video_ids[idx])
        clusters[label].descriptions.append(descriptions[idx])

    # Compute centroids
    for label, cluster in clusters.items():
        idxs = [i for i, l in enumerate(labels) if l == label]
        cluster.centroid = embeddings[idxs].mean(axis=0)

    return list(clusters.values())
