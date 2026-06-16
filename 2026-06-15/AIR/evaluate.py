"""
AIR - Evaluation Script

Computes NDCG@K, Hit Rate@K for cross-domain recommendation.
"""
import json
import math
import sys
from typing import Union


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Compute NDCG@k for a single user."""
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 2)

    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_set))))
    return dcg / (ideal_dcg + 1e-8)


def hit_rate_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Compute Hit Rate@k for a single user."""
    relevant_set = set(relevant)
    return float(any(item in relevant_set for item in recommended[:k]))


def evaluate(predictions_path: str, ground_truth_path: str, k: int = 10):
    """Main evaluation loop."""
    with open(predictions_path) as f:
        predictions: dict[str, list[str]] = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth: dict[str, list[str]] = json.load(f)

    ndcg_scores = []
    hr_scores = []
    n_evaluated = 0

    for uid, gt_items in ground_truth.items():
        if uid not in predictions or not gt_items:
            continue
        preds = predictions[uid]
        ndcg_scores.append(ndcg_at_k(preds, gt_items, k))
        hr_scores.append(hit_rate_at_k(preds, gt_items, k))
        n_evaluated += 1

    if n_evaluated == 0:
        print("No users evaluated.")
        return

    print(f"Evaluated {n_evaluated} users @ K={k}")
    print(f"  NDCG@{k}:      {sum(ndcg_scores)/len(ndcg_scores):.4f}")
    print(f"  HitRate@{k}:   {sum(hr_scores)/len(hr_scores):.4f}")


if __name__ == "__main__":
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "data/predictions.json"
    gt_path   = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.json"
    k         = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    evaluate(pred_path, gt_path, k)
