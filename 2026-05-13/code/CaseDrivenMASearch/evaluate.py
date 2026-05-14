"""
Evaluation script for GRM annotation quality and relevance accuracy.

Metrics:
    - Annotation Precision: % of agent-labeled samples matching gold
    - Label distribution
    - Cost reduction estimate (vs. human annotation baseline)

Usage:
    python evaluate.py --pred_file outputs/annotations_round3.jsonl \
                       --gold_file data/toy_ecom_pairs.jsonl
"""

import argparse
import json
from collections import Counter
from typing import List


LABEL_SPACE = ["exact", "substitute", "complement", "irrelevant"]

# Paper-reported baselines (§4 of the paper)
HUMAN_ANNOTATOR_PRECISION = 0.872   # baseline human precision
HUMAN_COST_PER_SAMPLE_USD = 0.05    # baseline human annotation cost per sample
LLM_COST_PER_SAMPLE_USD = 0.0123   # estimated LLM annotation cost (75.4% reduction)


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_precision(preds: List[dict], golds: List[dict]) -> float:
    gold_map = {(g["query"], g["product_title"]): g.get("label") for g in golds}
    correct = 0
    total = 0
    for p in preds:
        key = (p.get("query"), p.get("product_title"))
        gold = gold_map.get(key)
        if gold and p.get("final_label") == gold:
            correct += 1
        if gold:
            total += 1
    return correct / max(total, 1)


def compute_label_distribution(items: List[dict], label_key: str = "final_label") -> dict:
    counter = Counter(item.get(label_key) for item in items if item.get(label_key))
    total = sum(counter.values())
    return {k: {"count": v, "pct": v / total} for k, v in counter.items()}


def estimate_cost_savings(n_samples: int) -> dict:
    human_cost = n_samples * HUMAN_COST_PER_SAMPLE_USD
    llm_cost = n_samples * LLM_COST_PER_SAMPLE_USD
    reduction = (human_cost - llm_cost) / human_cost
    return {
        "n_samples": n_samples,
        "human_cost_usd": human_cost,
        "llm_cost_usd": llm_cost,
        "cost_reduction_pct": reduction * 100,
    }


def evaluate(pred_file: str, gold_file: str):
    preds = load_jsonl(pred_file)
    golds = load_jsonl(gold_file)

    print(f"\n{'='*50}")
    print("  ANNOTATION QUALITY EVALUATION")
    print(f"{'='*50}")

    precision = compute_precision(preds, golds)
    print(f"\n[Annotation Precision]")
    print(f"  Agent Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Human Baseline:  {HUMAN_ANNOTATOR_PRECISION:.4f} ({HUMAN_ANNOTATOR_PRECISION*100:.2f}%)")
    delta = precision - HUMAN_ANNOTATOR_PRECISION
    print(f"  Delta vs Human:  {delta:+.4f} ({delta*100:+.2f}%)")
    print(f"  Paper reports:   +2.4% improvement over human")

    print(f"\n[Label Distribution (Agent Predictions)]")
    dist = compute_label_distribution(preds)
    for label in LABEL_SPACE:
        if label in dist:
            d = dist[label]
            print(f"  {label:15s}: {d['count']:3d} ({d['pct']*100:.1f}%)")

    print(f"\n[Cost Estimation]")
    cost = estimate_cost_savings(len(preds))
    print(f"  Samples annotated:   {cost['n_samples']}")
    print(f"  Human cost:          ${cost['human_cost_usd']:.2f}")
    print(f"  LLM agent cost:      ${cost['llm_cost_usd']:.2f}")
    print(f"  Cost reduction:      {cost['cost_reduction_pct']:.1f}%")
    print(f"  Paper reports:       75.4% cost reduction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--gold_file", required=True)
    args = parser.parse_args()
    evaluate(args.pred_file, args.gold_file)
