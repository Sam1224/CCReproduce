from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_search_benchmark, encode_image_attributes, pad_title_tokens
from model import SAMD2QModel, expansion_tokens_from_logits

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def search_score(query_terms: list[str], indexed_terms: list[str]) -> float:
    overlap = len(set(query_terms).intersection(indexed_terms))
    return float(overlap)


def evaluate_catalog(model: SAMD2QModel) -> dict:
    benchmark = build_search_benchmark()
    baseline_hits = 0
    augmented_hits = 0
    baseline_quality = 0.0
    augmented_quality = 0.0
    baseline_zero = 0
    augmented_zero = 0

    products = [item["product"] for item in benchmark]
    for item in benchmark:
        target = item["product"]
        query_terms = item["query_terms"]

        title_ids = pad_title_tokens(target.title_tokens).unsqueeze(0).to(DEVICE)
        image_features = encode_image_attributes(target.visual_attributes).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(title_ids, image_features)
        predicted_tokens = expansion_tokens_from_logits(logits, k=2)[0]

        baseline_scores = []
        augmented_scores = []
        for product in products:
            base_terms = list(dict.fromkeys(product.title_tokens))
            aug_terms = list(dict.fromkeys(product.title_tokens + (predicted_tokens if product == target else [])))
            baseline_scores.append((search_score(query_terms, base_terms), product.business_value, product))
            augmented_scores.append((search_score(query_terms, aug_terms), product.business_value, product))

        baseline_ranked = sorted(baseline_scores, key=lambda row: (row[0], row[1]), reverse=True)
        augmented_ranked = sorted(augmented_scores, key=lambda row: (row[0], row[1]), reverse=True)

        if baseline_ranked[0][0] == 0:
            baseline_zero += 1
        if augmented_ranked[0][0] == 0:
            augmented_zero += 1

        if any(row[2] == target for row in baseline_ranked[:5]):
            baseline_hits += 1
            baseline_quality += target.business_value
        if any(row[2] == target for row in augmented_ranked[:5]):
            augmented_hits += 1
            augmented_quality += target.business_value

    total = len(benchmark)
    metrics = {
        "baseline_recall_at_5": round(baseline_hits / total, 4),
        "augmented_recall_at_5": round(augmented_hits / total, 4),
        "baseline_quality": round(baseline_quality / total, 4),
        "augmented_quality": round(augmented_quality / total, 4),
        "baseline_zero_result_rate": round(baseline_zero / total, 4),
        "augmented_zero_result_rate": round(augmented_zero / total, 4),
    }
    metrics["recall_gain"] = round(metrics["augmented_recall_at_5"] - metrics["baseline_recall_at_5"], 4)
    metrics["quality_gain"] = round(metrics["augmented_quality"] - metrics["baseline_quality"], 4)
    return metrics


def main() -> None:
    model = SAMD2QModel(hidden_dim=80).to(DEVICE)
    state = torch.load(ROOT / "sam_d2q.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    metrics = evaluate_catalog(model)
    out_path = ROOT / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
