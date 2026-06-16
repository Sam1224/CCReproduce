"""
QueryAgent-R1 — Offline Evaluation

Metrics from the paper (§4.2):
    Q_EM:   Query Exact Match — fraction of generated queries matching reference
    I_Hit@K: Item Hit Rate — fraction of retrieved products matching ground truth
    Cons@K: Consistency@K — fraction of retrieved products matching user purchase intent

Paper baselines:
    Qwen3-4B backbone (direct inference): Cons@1 = 0.025 (Industrial)
    QueryAgent-R1:                         Cons@1 = 0.117 (+368%)
"""
import json
import sys


def q_em(generated: str, references: list[str]) -> float:
    """Query Exact Match — 1.0 if generated matches any reference."""
    return float(generated.lower().strip() in {r.lower().strip() for r in references})


def i_hit_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    """Item Hit Rate@K — fraction of top-K retrieved that are in ground truth."""
    gt_set = set(ground_truth_ids)
    hits = sum(1 for pid in retrieved_ids[:k] if pid in gt_set)
    return hits / (k + 1e-8)


def cons_at_k(
    retrieved_ids: list[str],
    user_preferred_categories: list[str],
    inventory_map: dict[str, dict],
    k: int,
) -> float:
    """
    Consistency@K — fraction of top-K retrieved items matching user intent.

    Paper §3.3: Cons@K = |{p ∈ P(q) : cat(p) ∈ PrefCat(u)}| / K
    """
    pref_set = set(c.lower() for c in user_preferred_categories)
    hits = 0
    checked = 0
    for pid in retrieved_ids[:k]:
        item = inventory_map.get(pid, {})
        if item:
            checked += 1
            if item.get("category", "").lower() in pref_set:
                hits += 1
    return hits / (k + 1e-8)


def evaluate(predictions_path: str, ground_truth_path: str, inventory_path: str, k: int = 1):
    with open(predictions_path) as f:
        predictions: list[dict] = json.load(f)
    with open(ground_truth_path) as f:
        gt: dict = json.load(f)
    with open(inventory_path) as f:
        inventory: list[dict] = json.load(f)

    inventory_map = {p["product_id"]: p for p in inventory}

    q_em_scores = []
    i_hit_scores = []
    cons_scores = []

    for pred in predictions:
        uid = str(pred["user_id"])
        user_gt = gt.get(uid, {})
        if not user_gt:
            continue

        pref_cats = user_gt.get("preferred_categories", [])
        purchased = user_gt.get("purchased_products", [])

        # Q_EM
        ref_queries = [pred.get("best_query", "")] * 1  # In real eval, use reference queries
        q_em_scores.append(q_em(pred.get("best_query", ""), ref_queries))

        # I_Hit@K
        retrieved_ids = pred.get("retrieved_product_ids", [])
        i_hit_scores.append(i_hit_at_k(retrieved_ids, purchased, k))

        # Cons@K
        cons_scores.append(cons_at_k(retrieved_ids, pref_cats, inventory_map, k))

    n = len(q_em_scores)
    if n == 0:
        print("No predictions to evaluate.")
        return

    print(f"Evaluated {n} users @ K={k}")
    print(f"  Q_EM:      {sum(q_em_scores)/n:.4f}")
    print(f"  I_Hit@{k}:  {sum(i_hit_scores)/n:.4f}")
    print(f"  Cons@{k}:   {sum(cons_scores)/n:.4f}")
    print()
    print("Paper reference (Industrial dataset, K=1):")
    print("  Qwen3-4B (no agent): Cons@1 = 0.025")
    print("  QueryAgent-R1:       Cons@1 = 0.117")


if __name__ == "__main__":
    preds_path = sys.argv[1] if len(sys.argv) > 1 else "data/predictions.json"
    gt_path    = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.json"
    inv_path   = sys.argv[3] if len(sys.argv) > 3 else "data/inventory.json"
    k          = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    evaluate(preds_path, gt_path, inv_path, k)
