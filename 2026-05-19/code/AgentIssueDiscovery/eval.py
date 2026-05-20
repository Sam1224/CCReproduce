"""Evaluation script for AgentIssueDiscovery.

Metrics:
- Recall precision/recall/F1 for the recall stage
- Cluster purity for coarse clusters
- New issue detection rate (correctly identified new vs. variant clusters)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from agent.recall_agent import RecallAgent
from clustering.coarse_cluster import coarse_cluster
from clustering.embedder import VideoEmbedder
from clustering.fine_cluster import classify_clusters
from data.synthetic_videos import generate_synthetic_corpus, SyntheticVideo


def evaluate_recall_stage(corpus: list[SyntheticVideo]) -> dict:
    agent = RecallAgent(use_llm=False)
    recall_results = agent.batch_recall(corpus)

    # Ground truth: emerging issues should be recalled
    gt = [1 if (v.issue_type and "emerging" in v.issue_type) else 0 for v in corpus]
    pred = [1 if r.recalled else 0 for r in recall_results]

    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)

    return {
        "stage": "recall",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_recalled": sum(pred),
        "num_emerging": sum(gt),
    }


def evaluate_clustering(corpus: list[SyntheticVideo]) -> dict:
    agent = RecallAgent(use_llm=False)
    recall_results = agent.batch_recall(corpus)

    recalled_ids = {r.video_id for r in recall_results if r.recalled}
    recalled = [v for v in corpus if v.video_id in recalled_ids]

    if len(recalled) < 4:
        return {"error": "Too few recalled videos to cluster"}

    descriptions = [v.description + " " + " ".join(v.visual_tags) for v in recalled]
    embedder = VideoEmbedder(n_components=min(32, len(recalled) - 1))
    embeddings = embedder.fit_transform(descriptions)
    embeddings_map = {v.video_id: embeddings[i] for i, v in enumerate(recalled)}

    coarse_clusters = coarse_cluster(
        video_ids=[v.video_id for v in recalled],
        descriptions=descriptions,
        embeddings=embeddings,
        n_clusters=max(3, len(recalled) // 8),
    )

    fine_clusters = classify_clusters(coarse_clusters, embeddings_map)

    # Evaluate: among recalled emerging videos, what fraction are correctly
    # placed in "new_issue" clusters?
    new_issue_cluster_ids = {vid for c in fine_clusters if c.is_new_issue for vid in c.video_ids}
    true_new_vids = {v.video_id for v in recalled if v.issue_type == "emerging_new"}
    true_variant_vids = {v.video_id for v in recalled if v.issue_type == "emerging_variant"}

    correct_new = len(new_issue_cluster_ids & true_new_vids)
    total_new = max(len(true_new_vids), 1)

    new_issue_detection_rate = correct_new / total_new

    return {
        "stage": "clustering",
        "num_coarse_clusters": len(coarse_clusters),
        "num_fine_clusters": len(fine_clusters),
        "num_new_issue_clusters": sum(1 for c in fine_clusters if c.is_new_issue),
        "num_variant_clusters": sum(1 for c in fine_clusters if not c.is_new_issue),
        "new_issue_detection_rate": round(new_issue_detection_rate, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_videos", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n=== AgentIssueDiscovery Evaluation (N={args.num_videos}) ===\n")

    corpus = generate_synthetic_corpus(num_videos=args.num_videos, seed=args.seed)

    recall_metrics = evaluate_recall_stage(corpus)
    print("Recall Stage:")
    for k, v in recall_metrics.items():
        print(f"  {k}: {v}")

    clustering_metrics = evaluate_clustering(corpus)
    print("\nClustering Stage:")
    for k, v in clustering_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
