"""
QueryAgent-R1 — Memory-Augmented Query Generation Agent

Implements the core agent that generates e-commerce query recommendations
using chain-of-retrieval optimization.

Paper (§3.1): The agent maintains a short-term memory of retrieved products
from previous query iterations and uses this to refine subsequent queries.

Architecture:
    Policy network (LLM backbone) + Memory buffer
    Chain-of-retrieval loop: Generate → Retrieve → Refine → Repeat
"""
import json
import random
from typing import Optional

import numpy as np


CANDIDATE_QUERIES = {
    "laptop": [
        "best laptop for developers 2026",
        "lightweight ultrabook under 1kg",
        "gaming laptop with rtx 5090",
        "budget laptop student",
        "MacBook alternative laptop",
    ],
    "headphones": [
        "wireless headphones noise cancelling",
        "best audiophile headphones",
        "earbuds for gym workout",
        "headset for video calls",
        "open back headphones music",
    ],
    "sneakers": [
        "running sneakers for marathon",
        "casual white sneakers men",
        "basketball shoes high top",
        "trail running shoes waterproof",
        "minimalist barefoot sneakers",
    ],
    "coffee_maker": [
        "best espresso machine home",
        "drip coffee maker 12 cup",
        "cold brew coffee maker",
        "automatic cappuccino machine",
        "pour over coffee kettle set",
    ],
    "skincare_serum": [
        "vitamin c brightening serum",
        "anti-aging retinol serum",
        "hyaluronic acid moisturizing serum",
        "niacinamide pore minimizing serum",
        "peptide collagen boosting serum",
    ],
}

DEFAULT_QUERIES = [
    "best products 2026",
    "top rated items",
    "popular products",
    "highly recommended",
    "bestseller products",
]


class QueryGenerationAgent:
    """
    Memory-augmented agent for e-commerce query recommendation.

    Paper §3.1: The agent policy π(q|h_u, m_t) generates queries conditioned on:
        h_u = user interaction history
        m_t = short-term retrieval memory at step t

    The chain-of-retrieval loop (§3.2):
        for t in 1..T:
            q_t = π(·|h_u, m_t)
            p_t = Retrieve(q_t, Inventory)
            r_t = Reward(q_t, p_t, u)
            m_{t+1} = Update(m_t, q_t, p_t, r_t)
    """

    def __init__(
        self,
        retriever,
        reward_fn,
        max_iterations: int = 3,
        memory_size: int = 5,
    ):
        self.retriever = retriever
        self.reward_fn = reward_fn
        self.max_iterations = max_iterations
        self.memory = []  # (query, retrieved_products, reward) triples
        self.memory_size = memory_size

    def _mock_policy(
        self,
        user_history: dict,
        memory: list[dict],
        iteration: int,
    ) -> str:
        """
        Simulated policy network (LLM backbone).

        In production: a fine-tuned LLM (e.g., Qwen3-4B backbone) conditioned
        on user history tokens + memory summary.

        This toy policy:
        - t=0: generates initial query based on user preferences
        - t>0: refines based on retrieval feedback (consistency score)
        """
        preferred_cats = user_history.get("preferred_categories", [])

        if iteration == 0 or not memory:
            # Initial query generation
            for cat in preferred_cats:
                if cat in CANDIDATE_QUERIES:
                    return random.choice(CANDIDATE_QUERIES[cat])
            return random.choice(DEFAULT_QUERIES)
        else:
            # Refinement based on memory (chain-of-retrieval feedback)
            last_entry = memory[-1]
            last_cons = last_entry.get("reward", {}).get("r_cons", 0.0)

            if last_cons < 0.3:
                # Low consistency → switch to more specific query
                for cat in preferred_cats:
                    if cat in CANDIDATE_QUERIES:
                        queries = CANDIDATE_QUERIES[cat]
                        used = {e["query"] for e in memory}
                        unused = [q for q in queries if q not in used]
                        if unused:
                            return random.choice(unused)

            # Good consistency → minor refinement (add specificity)
            last_query = last_entry["query"]
            return last_query + " " + random.choice(["2026", "under budget", "highly rated"])

    def generate(self, user_history: dict) -> dict:
        """
        Run the chain-of-retrieval loop for a user.

        Returns the best query (highest consistency reward) and its metadata.
        """
        self.memory = []
        best_query = None
        best_reward = -1.0
        best_retrieved = []

        preferred_cats = user_history.get("preferred_categories", [])
        ref_queries = [
            q
            for cat in preferred_cats
            for q in CANDIDATE_QUERIES.get(cat, [])[:2]
        ] or DEFAULT_QUERIES

        for t in range(self.max_iterations):
            # Step 1: Generate query
            query = self._mock_policy(user_history, self.memory, iteration=t)

            # Step 2: Retrieve products (Chain-of-Retrieval)
            retrieved = self.retriever.retrieve(query, top_k=10)

            # Step 3: Compute reward
            reward = self.reward_fn(
                generated_query=query,
                reference_queries=ref_queries,
                retrieved_products=retrieved,
                user_preferred_categories=preferred_cats,
            )

            # Step 4: Update memory
            self.memory.append({
                "query": query,
                "retrieved": [p["product_id"] for p in retrieved[:5]],
                "reward": reward,
                "iteration": t,
            })
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

            # Track best
            if reward["total"] > best_reward:
                best_reward = reward["total"]
                best_query = query
                best_retrieved = retrieved

        return {
            "user_id": user_history["user_id"],
            "best_query": best_query,
            "best_reward": best_reward,
            "retrieved_product_ids": [p["product_id"] for p in best_retrieved[:20]],
            "iterations": len(self.memory),
        }


if __name__ == "__main__":
    from retriever import ProductRetriever
    from reward import ConsistencyReward

    with open("data/inventory.json") as f:
        inventory = json.load(f)
    with open("data/users.json") as f:
        users = json.load(f)

    retriever = ProductRetriever(inventory)
    reward_fn = ConsistencyReward(lambda_rel=0.4, top_k=10)
    agent = QueryGenerationAgent(retriever, reward_fn, max_iterations=3)

    # Test on first user
    user = users[0]
    result = agent.generate(user)
    print(f"User {result['user_id']}: best query = '{result['best_query']}'")
    print(f"  Best reward: {result['best_reward']:.4f}")
    print(f"  Top retrieved: {result['retrieved_product_ids'][:5]}")
