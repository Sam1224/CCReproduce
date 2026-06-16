"""
QueryAgent-R1 — Consistency Reward

Implements the joint reward function that optimizes both
query relevance (CTR) and product consistency (CVR).

Paper (§3.3 - Consistency Reward):
    R(q, u) = λ · R_rel(q, h_u) + (1-λ) · R_cons(q, u)

where:
    R_rel  = query-level relevance (BLEU / embedding similarity to reference queries)
    R_cons = Cons@K = fraction of retrieved products matching user intent
    λ      = trade-off weight (paper: λ=0.4)
"""
import math
from typing import Optional

import numpy as np


class ConsistencyReward:
    """
    Joint query relevance + product consistency reward for RL training.

    Paper §3.3: The key innovation is combining R_rel (query quality)
    with R_cons (retrieval-product alignment) so the agent optimizes
    not just CTR-proxy but also downstream CVR-proxy.
    """

    def __init__(self, lambda_rel: float = 0.4, top_k: int = 10):
        self.lambda_rel = lambda_rel
        self.top_k = top_k

    def relevance_reward(
        self,
        generated_query: str,
        reference_queries: list[str],
    ) -> float:
        """
        Query-level relevance reward R_rel.

        Approximated here with token-overlap (BLEU-1 unigram precision).
        In production: embedding cosine similarity or click-based reward.

        Paper uses: R_rel(q) ≈ E[CTR(q)]
        """
        gen_tokens = set(generated_query.lower().split())
        if not gen_tokens:
            return 0.0

        max_overlap = 0.0
        for ref in reference_queries:
            ref_tokens = set(ref.lower().split())
            if not ref_tokens:
                continue
            overlap = len(gen_tokens & ref_tokens) / len(gen_tokens)
            max_overlap = max(max_overlap, overlap)
        return max_overlap

    def consistency_reward(
        self,
        retrieved_products: list[dict],
        user_preferred_categories: list[str],
    ) -> float:
        """
        Product consistency reward R_cons.

        Paper formula (§3.3, Eq. 4):
            R_cons = |{p ∈ P(q) : cat(p) ∈ PrefCat(u)}| / K
        """
        pref_set = set(c.lower() for c in user_preferred_categories)
        k = min(self.top_k, len(retrieved_products))
        if k == 0:
            return 0.0
        matches = sum(1 for p in retrieved_products[:k] if p["category"].lower() in pref_set)
        return matches / k

    def __call__(
        self,
        generated_query: str,
        reference_queries: list[str],
        retrieved_products: list[dict],
        user_preferred_categories: list[str],
    ) -> dict:
        """
        Compute the joint reward.

        Returns: dict with reward components and total.
        """
        r_rel = self.relevance_reward(generated_query, reference_queries)
        r_cons = self.consistency_reward(retrieved_products, user_preferred_categories)

        # Joint reward (paper §3.3, Eq. 5)
        total = self.lambda_rel * r_rel + (1 - self.lambda_rel) * r_cons

        return {
            "r_rel": r_rel,
            "r_cons": r_cons,
            "total": total,
            "lambda_rel": self.lambda_rel,
        }


if __name__ == "__main__":
    reward_fn = ConsistencyReward(lambda_rel=0.4, top_k=10)

    # Example
    gen_query = "wireless noise cancelling headphones"
    ref_queries = ["best headphones", "wireless earbuds noise cancelling"]
    retrieved = [
        {"product_id": "p001", "category": "headphones", "title": "Premium Headphones"},
        {"product_id": "p002", "category": "laptop", "title": "Ultrabook Pro"},
        {"product_id": "p003", "category": "headphones", "title": "Sony WH1000XM6"},
        {"product_id": "p004", "category": "smartphone", "title": "iPhone 18"},
        {"product_id": "p005", "category": "headphones", "title": "Bose QC45"},
    ]
    user_prefs = ["headphones", "keyboard"]

    reward = reward_fn(gen_query, ref_queries, retrieved, user_prefs)
    print(f"R_rel:  {reward['r_rel']:.4f}")
    print(f"R_cons: {reward['r_cons']:.4f}")
    print(f"Total:  {reward['total']:.4f}")
