"""
Interactive demo for QueryAgent-R1.
Shows the chain-of-retrieval loop: user history → generated queries → retrieved products.
"""

import argparse
import torch
import numpy as np

from data import build_datasets
from model import QueryAgentR1
from train import sample_history_tensors


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog, users, _, _ = build_datasets(
        n_products=args.n_products,
        n_users=args.n_users,
        n_train=0,
        n_eval=1,
        embedding_dim=args.emb_dim,
        seed=args.seed,
    )

    model = QueryAgentR1(
        catalog_emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_refinement_steps=args.n_refine,
    ).to(device)

    user_id = args.user_id % len(users)
    user = users.get_user(user_id)
    if user:
        print(f"\nUser {user_id}")
        print(f"  Click history (first 5 product IDs): {user.clicked_product_ids[:5]}")
        print(f"  Purchase history: {user.purchased_product_ids[:3]}")
        print(f"  Query history: {user.query_history[:3]}")

    click_emb, purch_emb = sample_history_tensors(
        type("S", (), {"user_id": user_id})(), users, catalog, device
    )
    click_emb = click_emb.unsqueeze(0)
    purch_emb = purch_emb.unsqueeze(0)

    print("\n--- Chain-of-Retrieval Loop ---")
    with torch.no_grad():
        out = model(click_emb, purch_emb, catalog, temperature=1.0)

    print(f"\nGenerated query token ids: {out['query_ids_final'][0].tolist()[:10]}...")
    print(f"Query embedding norm: {out['query_emb_final'].norm().item():.3f}")
    print(f"Consistency Reward: {out['reward'][0].item():.4f}")

    print("\nTop-5 Retrieved Products:")
    for i, prod in enumerate(out["retrieved_products"][0]):
        print(f"  [{i+1}] {prod.title} | {prod.category} | ${prod.price:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, default=42)
    parser.add_argument("--n_products", type=int, default=500)
    parser.add_argument("--n_users", type=int, default=100)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_refine", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_demo(args)
