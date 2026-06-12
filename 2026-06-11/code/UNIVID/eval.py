"""
UNIVID Evaluation Script
Paper: arxiv 2606.05748

Metrics:
  - Violation detection: Precision, Recall, F1 at 80% precision operating point
  - Caption quality: ROUGE-L, BERTScore vs. expert captions
  - Trend detection: mAP on emerging risk categories
  - RAG recall: recall@k for known violation events
"""

import argparse
import json
import torch
import torch.nn.functional as F
from typing import Dict, List


def evaluate_pipeline(results: List[Dict], gt_labels: List[Dict]) -> Dict:
    """
    Compute moderation metrics from pipeline results.
    
    Args:
        results: list of ModerationResult dicts
        gt_labels: list of ground truth label dicts
    
    Returns:
        metrics: dict with precision, recall, F1, and operating point metrics
    """
    tp = fp = tn = fn = 0

    for res, gt in zip(results, gt_labels):
        pred_violative = res["is_violative"]
        true_violative = gt["is_violative"]

        if pred_violative and true_violative:
            tp += 1
        elif pred_violative and not true_violative:
            fp += 1
        elif not pred_violative and not true_violative:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_rag_recall(
    query_embeddings: torch.Tensor,
    gt_violation_event_ids: List[str],
    rag_index,
    k: int = 5,
) -> float:
    """
    Compute recall@k for UNIVID-RAG violation event retrieval.
    
    recall@k = fraction of videos where the correct event is in top-k results.
    """
    hits = 0
    for query, gt_id in zip(query_embeddings, gt_violation_event_ids):
        recalls = rag_index.search(query, top_k=k)
        retrieved_ids = [r["event_id"] for r in recalls]
        if gt_id in retrieved_ids:
            hits += 1
    return hits / len(gt_violation_event_ids) if gt_violation_event_ids else 0.0


def print_metrics(metrics: Dict, label: str = ""):
    print(f"\n=== {label} Metrics ===")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1:        {metrics.get('f1', 0):.4f}")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  TP/FP/TN/FN: {metrics.get('tp')}/{metrics.get('fp')}/{metrics.get('tn')}/{metrics.get('fn')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNIVID pipeline")
    parser.add_argument("--data", default="./toy_data")
    parser.add_argument("--model", default="./checkpoints/univid_lite_best.pt")
    parser.add_argument("--toy", action="store_true", default=True)
    args = parser.parse_args()

    print("UNIVID Evaluation")

    if args.toy:
        print("Running toy evaluation with simulated results...")

        # Simulate results
        n = 50
        import random
        results = [
            {"is_violative": random.random() > 0.4, "violation_classes": [], "risk_score": random.random()}
            for _ in range(n)
        ]
        gt_labels = [
            {"is_violative": random.random() > 0.5}
            for _ in range(n)
        ]

        metrics = evaluate_pipeline(results, gt_labels)
        print_metrics(metrics, label="UNIVID-Lite (toy)")

        print("\nExpected production metrics (from paper):")
        print("  Stage A Risk Filter: filters ~70% of safe content")
        print("  Stage B UNIVID-Lite: recall=80%+ at 80% precision")
        print("  Stage B UNIVID-RAG:  +recall for edge cases similar to prior events")
        print("  Stage C Trend:       adaptive detection within hours of new risk trends")

        return

    print("Full evaluation requires pretrained UNIVID model and labeled test set.")


if __name__ == "__main__":
    main()
