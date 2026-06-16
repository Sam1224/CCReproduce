"""
QueryAgent-R1 — RL Training Loop

Implements REINFORCE-based training of the query generation policy
using the consistency reward.

Paper (§3.4): The agent is trained with policy gradient using
the joint consistency reward R(q, u) = λ·R_rel + (1-λ)·R_cons.

Training objective:
    ∇_θ J(θ) = E_{q ~ π_θ}[R(q, u) · ∇_θ log π_θ(q|h_u)]
"""
import json
import os
import random
from typing import Optional

import numpy as np


class PolicyNetwork:
    """
    Toy policy network simulating the LLM backbone.

    In production (paper §3.4):
    - Backbone: Qwen3-4B (fine-tuned with RL)
    - Training: PPO / GRPO with consistency reward
    - Batch size: 64 users per step

    This toy: weight matrix mapping context → query score logits.
    """

    def __init__(self, n_categories: int = 25, n_query_templates: int = 50, lr: float = 1e-3):
        self.n_cats = n_categories
        self.n_templates = n_query_templates
        self.lr = lr
        # Simple linear policy: context vector → template logits
        self.W = np.random.randn(n_categories, n_query_templates) * 0.01

    def forward(self, context_vec: np.ndarray) -> np.ndarray:
        """Compute logits over query templates."""
        return context_vec @ self.W  # shape: [n_templates]

    def policy(self, context_vec: np.ndarray) -> np.ndarray:
        """Softmax distribution over query templates."""
        logits = self.forward(context_vec)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-8
        return probs

    def update(self, context_vec: np.ndarray, action: int, reward: float, baseline: float = 0.0):
        """
        REINFORCE gradient update.

        ∇_θ J(θ) ≈ (R - baseline) · ∇_θ log π_θ(a|s)
        """
        probs = self.policy(context_vec)
        # Gradient of log π(a|s) w.r.t. W
        advantage = reward - baseline
        one_hot = np.zeros(self.n_templates)
        one_hot[action] = 1.0
        grad_log_pi = one_hot - probs  # shape: [n_templates]
        # dW = context_vec^T · grad_log_pi^T
        self.W += self.lr * advantage * np.outer(context_vec, grad_log_pi)


def train(
    users_path: str = "data/users.json",
    inventory_path: str = "data/inventory.json",
    gt_path: str = "data/ground_truth.json",
    epochs: int = 5,
    output_dir: str = "checkpoints",
):
    """Main RL training loop."""
    os.makedirs(output_dir, exist_ok=True)

    from retriever import ProductRetriever
    from reward import ConsistencyReward
    from agent import QueryGenerationAgent, CANDIDATE_QUERIES, DEFAULT_QUERIES

    with open(users_path) as f:
        users = json.load(f)
    with open(inventory_path) as f:
        inventory = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    retriever = ProductRetriever(inventory)
    reward_fn = ConsistencyReward(lambda_rel=0.4, top_k=10)

    CATEGORIES = [
        "laptop", "headphones", "keyboard", "monitor", "smartphone",
        "sneakers", "backpack", "jacket", "sunglasses", "watch",
        "coffee_maker", "blender", "air_fryer", "cookware", "knife_set",
        "yoga_mat", "dumbbells", "protein_powder", "running_shoes", "resistance_band",
        "skincare_serum", "moisturizer", "sunscreen", "face_mask", "eye_cream",
    ]
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

    ALL_QUERIES = [q for qs in CANDIDATE_QUERIES.values() for q in qs] + DEFAULT_QUERIES
    policy = PolicyNetwork(n_categories=len(CATEGORIES), n_query_templates=len(ALL_QUERIES))

    running_reward = 0.0
    gamma = 0.99

    print(f"Training QueryAgent-R1 policy ({epochs} epochs, {len(users)} users)...")

    for epoch in range(epochs):
        epoch_rewards = []
        random.shuffle(users)

        for user in users[:100]:  # subsample for speed
            # Build context vector from user preferred categories
            ctx = np.zeros(len(CATEGORIES), dtype=np.float32)
            for cat in user.get("preferred_categories", []):
                if cat in cat_to_idx:
                    ctx[cat_to_idx[cat]] = 1.0

            # Sample action (query template)
            probs = policy.policy(ctx)
            action = np.random.choice(len(ALL_QUERIES), p=probs)
            query = ALL_QUERIES[action]

            # Chain-of-Retrieval: retrieve products
            retrieved = retriever.retrieve(query, top_k=10)

            # Compute reward
            ref_queries = [
                q for cat in user.get("preferred_categories", [])
                for q in CANDIDATE_QUERIES.get(cat, [])[:2]
            ] or DEFAULT_QUERIES
            reward = reward_fn(
                generated_query=query,
                reference_queries=ref_queries,
                retrieved_products=retrieved,
                user_preferred_categories=user.get("preferred_categories", []),
            )
            r = reward["total"]

            # REINFORCE update
            running_reward = gamma * running_reward + (1 - gamma) * r
            policy.update(ctx, action, r, baseline=running_reward)
            epoch_rewards.append(r)

        mean_r = sum(epoch_rewards) / (len(epoch_rewards) + 1e-8)
        print(f"Epoch {epoch+1}/{epochs}: mean_reward={mean_r:.4f}")

    # Save checkpoint (simplified)
    checkpoint = {
        "W_shape": list(policy.W.shape),
        "W_flat": policy.W.flatten().tolist(),
        "categories": CATEGORIES,
        "queries": ALL_QUERIES,
        "final_mean_reward": mean_r,
    }
    ckpt_path = os.path.join(output_dir, "best.json")
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f)
    print(f"\nSaved checkpoint to {ckpt_path}.")

    # Generate predictions for evaluation
    predictions = []
    agent = QueryGenerationAgent(retriever, reward_fn, max_iterations=3)
    for user in users[:50]:
        result = agent.generate(user)
        predictions.append(result)

    with open("data/predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions for {len(predictions)} users.")


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    train(epochs=epochs)
