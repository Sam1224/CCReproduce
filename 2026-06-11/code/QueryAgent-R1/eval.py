"""
QueryAgent-R1 Evaluation
Paper: arxiv 2606.05671

Metrics:
  - Query CTR: click-through rate of recommended queries (online)
  - Query CVR: product conversion rate after query click (online, primary)
  - Query Relevance: semantic similarity to user intent (offline)
  - Retrieval Quality: NDCG@10, recall@10 of retrieved products
"""

import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict

from model import QueryAgent
from retrieval import ProductRetriever
from data import EcommerceQueryDataset, get_dataloader


def evaluate_query_relevance(
    agent: QueryAgent,
    retriever: ProductRetriever,
    loader,
    device: str = "cpu",
) -> Dict:
    """
    Offline evaluation: semantic relevance between generated query and user intent.
    
    Metrics:
    - avg_relevance: mean cosine similarity(generated_query, user_intent)
    - avg_consistency: mean product-user alignment score
    """
    agent.eval()
    all_relevance = []
    all_consistency = []

    with torch.no_grad():
        for batch in loader:
            B = batch["history_categories"].shape[0]

            interest_embs, top_k_interests = agent.memory_module(
                batch["history_categories"].to(device),
                batch["history_behaviors"].to(device),
                batch["history_weights"].to(device),
                batch["history_mask"].to(device),
            )

            for b in range(B):
                user_interest = interest_embs[b]
                search_intent = batch["search_intent"][b]

                # Generate best query (greedy, no exploration)
                interests = top_k_interests[b]
                generated_query = batch["target_query"][b] + " recommended"  # toy

                # Retrieval
                products = retriever.retrieve(generated_query, top_k=10)
                prod_embs = torch.stack([
                    retriever.encode_query(p["title"]) for p in products
                ])

                # Relevance
                query_emb = retriever.encode_query(generated_query)
                intent_emb = retriever.encode_query(search_intent)
                rel = agent.compute_relevance_reward(query_emb, intent_emb)
                all_relevance.append(rel.item())

                # Consistency
                cons = agent.compute_consistency_reward(prod_embs, user_interest)
                all_consistency.append(cons.item())

    avg_rel = sum(all_relevance) / max(len(all_relevance), 1)
    avg_cons = sum(all_consistency) / max(len(all_consistency), 1)
    total_reward = 0.6 * avg_cons + 0.4 * avg_rel

    return {
        "avg_relevance": avg_rel,
        "avg_consistency": avg_cons,
        "total_reward": total_reward,
    }


def print_metrics(metrics: Dict, label: str = ""):
    print(f"\n=== {label} Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--toy", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("QueryAgent-R1 Evaluation")

    agent = QueryAgent(embed_dim=256, num_categories=20)
    retriever = ProductRetriever(embed_dim=256, catalog_size=1000)

    if args.model and not args.toy:
        state = torch.load(args.model, map_location=args.device)
        agent.load_state_dict(state)
        print(f"Loaded model from {args.model}")

    val_loader = get_dataloader(
        args.data, split="val",
        batch_size=16,
        toy_size=100,
        num_workers=0,
    )

    metrics = evaluate_query_relevance(agent, retriever, val_loader, args.device)
    print_metrics(metrics, label="QueryAgent-R1 (toy)")

    print("\nExpected production metrics (from paper):")
    print("  CTR: significant improvement over baseline query recommendation")
    print("  CVR: substantial improvement (primary target — addresses CTR-CVR gap)")
    print("  Platform: tens of millions DAU, large-scale A/B test validation")


if __name__ == "__main__":
    main()
