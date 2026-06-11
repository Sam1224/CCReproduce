"""
Offline evaluation for QueryAgent-R1.

Metrics (adapted from paper §4):
  - MeanReward:  average consistency reward (α·CTR_proxy + β·CVR_proxy)
  - CTR_proxy:   cosine sim(query_emb, history_emb)
  - CVR_proxy:   cosine sim(retrieved_emb, purchase_emb)
  - RetrievalDiversity: mean pairwise distance among retrieved products per query
  - QueryUniqueness: fraction of unique tokens in generated queries
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List

from data import build_datasets
from model import QueryAgentR1
from train import collate_batch


@torch.no_grad()
def run_eval(model: QueryAgentR1, eval_set, users, catalog, device, batch_size=32) -> dict:
    model.eval()
    metrics = {
        "mean_reward": [],
        "ctr_proxy": [],
        "cvr_proxy": [],
        "retrieval_diversity": [],
    }

    indices = list(range(len(eval_set)))
    for start in tqdm(range(0, len(indices), batch_size), desc="Evaluating"):
        batch_ids = indices[start:start + batch_size]
        batch_samples = [eval_set[i] for i in batch_ids]
        click_embs, purch_embs = collate_batch(batch_samples, users, catalog, device)

        out = model(click_embs, purch_embs, catalog)

        query_emb = out["query_emb_final"]           # (B, D)
        context, hist_emb, purch_emb = model.encode_history(click_embs, purch_embs)

        # Compute retrieved mean embeddings
        retrieved = out["retrieved_products"]
        ret_emb_list = []
        diversity_list = []
        for batch_results in retrieved:
            embs = np.stack([p.embedding for p in batch_results])  # (k, D)
            ret_emb_list.append(embs.mean(0))
            # Pairwise cosine distance (diversity)
            normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            sim_matrix = normed @ normed.T
            k = normed.shape[0]
            off_diag = sim_matrix[np.triu_indices(k, k=1)]
            diversity = 1.0 - off_diag.mean() if len(off_diag) > 0 else 0.0
            diversity_list.append(diversity)

        ret_emb = torch.tensor(np.stack(ret_emb_list), dtype=torch.float32, device=device)

        ctr = F.cosine_similarity(query_emb, hist_emb, dim=-1).cpu().numpy()
        cvr = F.cosine_similarity(ret_emb, purch_emb, dim=-1).cpu().numpy()
        reward = (0.5 * ctr + 0.5 * cvr)

        metrics["mean_reward"].extend(reward.tolist())
        metrics["ctr_proxy"].extend(ctr.tolist())
        metrics["cvr_proxy"].extend(cvr.tolist())
        metrics["retrieval_diversity"].extend(diversity_list)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate QueryAgent-R1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--n_products", type=int, default=2000)
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_refine", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, users, _, eval_set = build_datasets(
        n_products=args.n_products,
        n_users=args.n_users,
        n_train=0,
        n_eval=args.n_eval,
        embedding_dim=args.emb_dim,
        seed=args.seed,
    )

    model = QueryAgentR1(
        catalog_emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_refinement_steps=args.n_refine,
    ).to(device)

    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint}; using random weights.")

    results = run_eval(model, eval_set, users, catalog, device, args.batch_size)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
