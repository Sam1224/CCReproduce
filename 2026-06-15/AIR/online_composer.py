"""
AIR - Online Intent Composition

Simulates the online serving phase: retrieve relevant intent atoms for each
user × item pair and compose them into a unified intent representation for ranking.

Paper (§3.3): Online composition aggregates top-k retrieved atoms into a
single vector via attention-weighted pooling, then scores against item embeddings.

Key design: ~400× latency reduction vs. calling LLM online.
"""
import json
import os
from typing import Optional

import numpy as np

from atom_index import AtomIndex


class IntentComposer:
    """
    Online intent composition layer.

    Given a user's current context, retrieves their relevant intent atoms
    and composes them into a unified representation for cross-domain item ranking.

    Paper formulation (§3.3):
        r_u = Compose(Retrieve(q_u, A, k))
            = Σ_i α_i · e(a_i)       (attention-weighted sum)
        where α_i = softmax(score(q_u, a_i) / τ)
    """

    def __init__(self, atom_index: AtomIndex, top_k: int = 10, temperature: float = 0.1):
        self.atom_index = atom_index
        self.top_k = top_k
        self.temperature = temperature

    def _build_query(self, context: dict) -> str:
        """Build retrieval query from user context."""
        parts = []
        if "recent_categories" in context:
            parts.append(f"interested in {', '.join(context['recent_categories'])}")
        if "current_item_category" in context:
            parts.append(f"browsing {context['current_item_category']}")
        return " ".join(parts) or "general user"

    def compose(self, context: dict) -> np.ndarray:
        """
        Retrieve and compose intent atoms into a user intent vector.

        Returns: composed_intent (shape: [embed_dim])
        """
        query = self._build_query(context)
        atoms = self.atom_index.search(query, top_k=self.top_k)

        if not atoms:
            return np.zeros(self.atom_index.embed_dim, dtype=np.float32)

        # Attention-weighted composition (paper §3.3, Eq. 3)
        scores = np.array([a["retrieval_score"] for a in atoms], dtype=np.float32)
        weights = np.exp(scores / self.temperature)
        weights /= weights.sum() + 1e-8

        # Get atom embeddings (re-encode for composability; in prod use cached embeddings)
        atom_vecs = np.stack([
            self.atom_index._encode_text(a["text"]) for a in atoms
        ])
        composed = (weights[:, None] * atom_vecs).sum(axis=0)
        composed /= np.linalg.norm(composed) + 1e-8
        return composed

    def score_items(self, context: dict, items: list[dict]) -> list[dict]:
        """
        Score e-commerce items for a user context using composed intent.

        Paper §3.4: ranking score = composed_intent · item_embedding
        """
        intent_vec = self.compose(context)

        scored_items = []
        for item in items:
            # In production, item embeddings are pre-computed in the item tower
            item_vec = self.atom_index._encode_text(
                item["title"] + " " + item["category"]
            )
            score = float(np.dot(intent_vec, item_vec))
            scored_items.append({**item, "score": score})

        return sorted(scored_items, key=lambda x: -x["score"])


def run_inference(
    index_path: str,
    items_path: str,
    output_path: str,
    n_users: int = 20,
):
    """Run online inference simulation over a set of users."""
    print(f"Loading atom index from {index_path}...")
    atom_idx = AtomIndex.load(index_path)

    with open(items_path) as f:
        items = json.load(f)

    composer = IntentComposer(atom_idx, top_k=10)

    predictions = {}
    categories = ["cooking", "fitness", "fashion", "travel", "gaming",
                  "beauty", "tech_review", "home_decor", "pets", "music"]

    import random
    for uid in range(n_users):
        context = {
            "user_id": uid,
            "recent_categories": random.sample(categories, k=3),
        }
        ranked = composer.score_items(context, items)
        predictions[str(uid)] = [it["item_id"] for it in ranked[:20]]

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions for {n_users} users to {output_path}.")
    return predictions


if __name__ == "__main__":
    import sys
    index_path = sys.argv[1] if len(sys.argv) > 1 else "data/atom_index"
    items_path = sys.argv[2] if len(sys.argv) > 2 else "data/ecom_items.json"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "data/predictions.json"

    run_inference(index_path, items_path, output_path)
