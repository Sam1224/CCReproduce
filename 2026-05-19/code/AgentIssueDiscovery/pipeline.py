"""End-to-end AgentIssueDiscovery pipeline.

Implements the 3-stage pipeline from "When Rules Fall Short":
  1. Agent Recall   — identify candidates with emerging issues
  2. Two-Stage Clustering — coarse → fine to separate variants vs. new issues
  3. Policy Generation — auto-generate annotation policies from clusters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent.recall_agent import RecallAgent
from clustering.coarse_cluster import coarse_cluster
from clustering.embedder import VideoEmbedder
from clustering.fine_cluster import classify_clusters
from data.synthetic_videos import generate_synthetic_corpus
from policy_gen.policy_generator import PolicyGenerator


def run_pipeline(
    num_videos: int = 500,
    output_dir: str = "runs/toy",
    emerging_ratio: float = 0.20,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Generate corpus ─────────────────────────────────────────────
    if verbose:
        print(f"\n[Pipeline] Generating {num_videos} synthetic videos...")
    corpus = generate_synthetic_corpus(num_videos=num_videos, emerging_ratio=emerging_ratio, seed=seed)
    if verbose:
        n_emerging = sum(1 for v in corpus if v.issue_type and "emerging" in v.issue_type)
        print(f"[Pipeline] Corpus: {len(corpus)} videos, {n_emerging} emerging issues")

    # ── Step 1: Agent Recall ─────────────────────────────────────────────────
    if verbose:
        print("\n[Pipeline] Stage 1: Agent Recall...")
    agent = RecallAgent(use_llm=False)
    recall_results = agent.batch_recall(corpus, verbose=verbose)

    recalled_video_ids = {r.video_id for r in recall_results if r.recalled}
    recalled_videos = [v for v in corpus if v.video_id in recalled_video_ids]
    if verbose:
        print(f"[Pipeline] Recalled {len(recalled_videos)} / {len(corpus)} videos")

    if len(recalled_videos) == 0:
        print("[Pipeline] No videos recalled. Exiting.")
        return {}

    # ── Step 2a: Embed recalled videos ───────────────────────────────────────
    if verbose:
        print("\n[Pipeline] Stage 2a: Embedding recalled videos...")
    descriptions = [v.description + " " + " ".join(v.visual_tags) for v in recalled_videos]
    embedder = VideoEmbedder(n_components=min(64, len(recalled_videos) - 1))
    embeddings = embedder.fit_transform(descriptions)
    embeddings_map = {v.video_id: embeddings[i] for i, v in enumerate(recalled_videos)}

    # ── Step 2b: Coarse clustering ───────────────────────────────────────────
    if verbose:
        print("[Pipeline] Stage 2b: Coarse clustering...")
    video_ids_list = [v.video_id for v in recalled_videos]
    coarse_clusters = coarse_cluster(
        video_ids=video_ids_list,
        descriptions=descriptions,
        embeddings=embeddings,
        n_clusters=max(3, len(recalled_videos) // 10),
    )
    if verbose:
        print(f"[Pipeline] Coarse clusters: {len(coarse_clusters)}")

    # ── Step 2c: Fine clustering ─────────────────────────────────────────────
    if verbose:
        print("[Pipeline] Stage 2c: Fine clustering...")
    fine_clusters = classify_clusters(coarse_clusters, embeddings_map)
    new_issue_clusters = [c for c in fine_clusters if c.is_new_issue]
    variant_clusters = [c for c in fine_clusters if not c.is_new_issue]
    if verbose:
        print(f"[Pipeline] Fine clusters: {len(fine_clusters)} total, "
              f"{len(new_issue_clusters)} new issues, {len(variant_clusters)} variants")

    # ── Step 3: Policy generation ─────────────────────────────────────────────
    if verbose:
        print("\n[Pipeline] Stage 3: Generating policies...")
    gen = PolicyGenerator(use_llm=False)
    policies = gen.generate_all(fine_clusters)
    new_policies = [p for p in policies if p.issue_name != "[Variant Update]"]
    if verbose:
        print(f"[Pipeline] Generated {len(new_policies)} new issue policies")
        for p in new_policies:
            print(f"  → [{p.severity}] {p.issue_name}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "num_videos": len(corpus),
        "num_recalled": len(recalled_videos),
        "num_coarse_clusters": len(coarse_clusters),
        "num_fine_clusters": len(fine_clusters),
        "num_new_issues": len(new_issue_clusters),
        "num_variants": len(variant_clusters),
        "generated_policies": [
            {
                "issue_name": p.issue_name,
                "description": p.description,
                "indicators": p.example_indicators,
                "severity": p.severity,
                "source_cluster": p.source_cluster_id,
            }
            for p in new_policies
        ],
    }

    out_file = output_path / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n[Pipeline] Results saved to {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AgentIssueDiscovery Pipeline")
    parser.add_argument("--num_videos", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="runs/toy")
    parser.add_argument("--emerging_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        num_videos=args.num_videos,
        output_dir=args.output_dir,
        emerging_ratio=args.emerging_ratio,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
