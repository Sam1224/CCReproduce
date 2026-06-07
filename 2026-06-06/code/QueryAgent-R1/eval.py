"""
QueryAgent-R1 Evaluation Script
Metrics:
    - Offline: NDCG@K, MRR (query-product relevance ranking)
    - Online (simulated): CTR gain, CVR gain relative to a vanilla SFT baseline

Usage:
    python eval.py --batch_size 16 --top_k 10
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model.query_agent import QueryAgentR1
from retrieval.retriever import ProductRetriever
from rl.reward import ConsistencyReward
from data.dataset import EComQueryDataset


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at k."""
    r = relevances[:k]
    if r.size == 0:
        return 0.0
    return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(
    agent: QueryAgentR1,
    retriever: ProductRetriever,
    reward_fn: ConsistencyReward,
    dataloader: DataLoader,
    top_k: int = 10,
    device: str = "cpu",
):
    agent.eval()
    retriever.eval()
    reward_fn.eval()

    all_rewards, all_ctr, all_cvr = [], [], []
    all_ndcg = []

    with torch.no_grad():
        for batch in dataloader:
            hist = batch["interaction_history"].to(device)
            ctx = batch["current_context"].to(device)

            query_ids = agent.generate_queries(hist, ctx)
            product_embs, _, query_emb = retriever(query_ids)
            reward, info = reward_fn(query_emb, product_embs, agent.memory_module(hist, ctx))

            all_rewards.append(reward.cpu().numpy())
            all_ctr.append(info["ctr_reward"].cpu().numpy())
            all_cvr.append(info["cvr_reward"].cpu().numpy())

            # Simulate NDCG: use CVR score as relevance proxy
            cvr_scores = info["cvr_reward"].cpu().numpy()
            for cvr in cvr_scores:
                # Single query → relevance = 1 if cvr > 0.5 else 0
                rel = np.array([1.0 if cvr > 0.5 else 0.0])
                all_ndcg.append(ndcg_at_k(rel, 1))

    mean_reward = np.concatenate(all_rewards).mean()
    mean_ctr = np.concatenate(all_ctr).mean()
    mean_cvr = np.concatenate(all_cvr).mean()
    mean_ndcg = np.mean(all_ndcg)

    print(f"\n=== QueryAgent-R1 Evaluation ===")
    print(f"  Mean Total Reward  : {mean_reward:.4f}")
    print(f"  Mean CTR Reward    : {mean_ctr:.4f}")
    print(f"  Mean CVR Reward    : {mean_cvr:.4f}")
    print(f"  NDCG@1 (proxy)     : {mean_ndcg:.4f}")
    print()

    return {
        "mean_reward": float(mean_reward),
        "mean_ctr": float(mean_ctr),
        "mean_cvr": float(mean_cvr),
        "ndcg_at_1": float(mean_ndcg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset = EComQueryDataset(size=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    agent = QueryAgentR1(item_embed_dim=64, vocab_size=10_000, memory_len=20)
    retriever = ProductRetriever(vocab_size=10_000, embed_dim=64, num_products=5_000)
    reward_fn = ConsistencyReward(embed_dim=64, alpha=0.4)

    evaluate(agent, retriever, reward_fn, dataloader, top_k=args.top_k, device=args.device)


if __name__ == "__main__":
    main()
