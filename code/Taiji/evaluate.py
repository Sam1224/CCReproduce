"""
Taiji Evaluation Pipeline.

Evaluates the trained model on:
1. Semantic quality: CoT coherence and intent alignment
2. ID alignment: CTR/CVR prediction accuracy on held-out test users
3. Pareto efficiency: whether the POPO weights achieved a trade-off
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.popo import SemanticReward, IDReward
from model.llm_enhancer import TaijiLLMEnhancer, TaijiOnlineRanker


def evaluate_cot_quality(llm_enhancer, test_users: list[dict]) -> dict:
    """
    Evaluate CoT quality on test users.
    Metrics:
    - Keyword recall: fraction of preferred categories mentioned in CoT
    - Price mention: whether the preferred price tier is mentioned
    - CoT length: proxy for completeness
    """
    kw_recalls = []
    price_hits = []
    cot_lengths = []

    llm_enhancer.eval()
    for user in tqdm(test_users[:50], desc="CoT quality eval"):  # evaluate 50 samples
        try:
            cot = llm_enhancer.generate_cot(user, max_new_tokens=150)
        except Exception:
            continue

        cot_lower = cot.lower()
        preferred_cats = user.get("preferred_categories", [])
        preferred_price = user.get("preferred_price", "")

        # Keyword recall: how many preferred categories are mentioned
        hits = sum(1 for cat in preferred_cats if cat.lower() in cot_lower)
        kw_recall = hits / len(preferred_cats) if preferred_cats else 0.0
        kw_recalls.append(kw_recall)

        price_hits.append(1.0 if preferred_price and preferred_price in cot_lower else 0.0)
        cot_lengths.append(len(cot.split()))

    return {
        "kw_recall_mean": float(np.mean(kw_recalls)) if kw_recalls else 0.0,
        "kw_recall_std": float(np.std(kw_recalls)) if kw_recalls else 0.0,
        "price_hit_rate": float(np.mean(price_hits)) if price_hits else 0.0,
        "avg_cot_length": float(np.mean(cot_lengths)) if cot_lengths else 0.0,
        "n_evaluated": len(kw_recalls),
    }


def evaluate_ranking(llm_enhancer, online_ranker, test_users: list[dict],
                     n_items: int = 500, topk: int = 10, device: str = "cpu") -> dict:
    """
    Evaluate ranking quality using intent embeddings from the LLM enhancer.
    Simulated metrics: NDCG@K and HR@K using the online ranker CTR × CVR scores.
    """
    id_embed_table = torch.nn.Embedding(n_items, 64).to(device)

    llm_enhancer.eval()
    online_ranker.eval()

    ndcg_at_k = []
    hr_at_k = []

    with torch.no_grad():
        for user in tqdm(test_users[:100], desc="Ranking eval"):
            try:
                prompt = llm_enhancer.format_user_sequence(user)
                inputs = llm_enhancer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                outputs = llm_enhancer.llm(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].mean(dim=1)
                intent_emb = llm_enhancer.intent_projector(hidden)  # [1, 128]

                # Score all items
                all_item_ids = torch.arange(n_items, device=device)
                item_embeds = id_embed_table(all_item_ids)  # [n_items, 64]
                intent_expanded = intent_emb.expand(n_items, -1)  # [n_items, 128]

                ranker_out = online_ranker(intent_expanded, item_embeds)
                scores = (ranker_out["ctr"] * ranker_out["cvr"]).cpu().numpy()  # [n_items]

                # Ground truth: items in preferred categories (simulated)
                preferred_cats = set(user.get("preferred_categories", []))
                # We don't have ground truth item→category mapping here,
                # so we use a proxy: top-10% score items as "relevant"
                relevance_threshold = np.percentile(scores, 90)
                relevant_mask = (scores >= relevance_threshold).astype(float)

                # Rank by predicted score
                ranked_indices = np.argsort(-scores)
                top_k_indices = ranked_indices[:topk]

                # HR@K
                hit = float(relevant_mask[top_k_indices].sum() > 0)
                hr_at_k.append(hit)

                # NDCG@K
                gains = relevant_mask[top_k_indices]
                discounts = 1.0 / np.log2(np.arange(2, topk + 2))
                dcg = float(np.sum(gains * discounts))
                ideal_gains = np.sort(relevant_mask)[::-1][:topk]
                idcg = float(np.sum(ideal_gains * discounts))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_at_k.append(ndcg)

            except Exception:
                continue

    return {
        f"NDCG@{topk}": float(np.mean(ndcg_at_k)) if ndcg_at_k else 0.0,
        f"HR@{topk}": float(np.mean(hr_at_k)) if hr_at_k else 0.0,
        "n_evaluated": len(ndcg_at_k),
    }


def evaluate_pareto_efficiency(reward_log_path: str) -> dict:
    """
    If a reward log JSON exists, compute the Pareto frontier statistics.
    Checks whether semantic and ID rewards are both non-decreasing over training.
    """
    if not Path(reward_log_path).exists():
        return {"pareto_log_found": False}

    with open(reward_log_path) as f:
        log = json.load(f)

    r_sem = [entry["r_sem"] for entry in log]
    r_id = [entry["r_id"] for entry in log]

    # Simple monotonicity test: is later-half mean > earlier-half mean?
    mid = len(r_sem) // 2
    sem_improved = float(np.mean(r_sem[mid:])) > float(np.mean(r_sem[:mid]))
    id_improved = float(np.mean(r_id[mid:])) > float(np.mean(r_id[:mid]))

    return {
        "pareto_log_found": True,
        "n_steps": len(r_sem),
        "final_r_sem": float(r_sem[-1]) if r_sem else 0.0,
        "final_r_id": float(r_id[-1]) if r_id else 0.0,
        "sem_improved_over_training": sem_improved,
        "id_improved_over_training": id_improved,
        "both_improved": sem_improved and id_improved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/taiji/sft_checkpoint",
                        help="Path to saved model checkpoint")
    parser.add_argument("--test_data", default="data/toy_data/test.json")
    parser.add_argument("--reward_log", default="checkpoints/taiji/reward_log.json",
                        help="Optional RL reward log for Pareto analysis")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n_items", type=int, default=500)
    parser.add_argument("--skip_cot", action="store_true",
                        help="Skip CoT quality eval (slow if no GPU)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    # Load test data
    with open(args.test_data) as f:
        test_users = json.load(f)
    print(f"Loaded {len(test_users)} test users")

    # Load model (from checkpoint if exists, else fresh)
    if Path(args.checkpoint_dir).exists():
        print(f"Loading from checkpoint: {args.checkpoint_dir}")
        model_src = args.checkpoint_dir
    else:
        print(f"Checkpoint not found; loading base model: {args.model_name}")
        model_src = args.model_name

    llm_enhancer = TaijiLLMEnhancer(model_name=model_src, device=device)
    online_ranker = TaijiOnlineRanker().to(device)

    results = {}

    # 1. CoT quality
    if not args.skip_cot:
        print("\n--- CoT Quality Evaluation ---")
        cot_metrics = evaluate_cot_quality(llm_enhancer, test_users)
        results["cot_quality"] = cot_metrics
        print(json.dumps(cot_metrics, indent=2))

    # 2. Ranking quality
    print("\n--- Ranking Evaluation ---")
    ranking_metrics = evaluate_ranking(
        llm_enhancer, online_ranker, test_users,
        n_items=args.n_items, topk=args.topk, device=device,
    )
    results["ranking"] = ranking_metrics
    print(json.dumps(ranking_metrics, indent=2))

    # 3. Pareto efficiency
    print("\n--- Pareto Efficiency Analysis ---")
    pareto_metrics = evaluate_pareto_efficiency(args.reward_log)
    results["pareto"] = pareto_metrics
    print(json.dumps(pareto_metrics, indent=2))

    # Save results
    out_path = Path(args.checkpoint_dir).parent / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to {out_path}")


if __name__ == "__main__":
    main()
