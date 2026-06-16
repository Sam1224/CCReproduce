"""
QueryAgent-R1 — Chain-of-Retrieval Module

Simulates the inventory retrieval API used during query validation.

Paper (§3.2): QueryAgent-R1 grounds query generation in real inventory
retrieval — the agent calls this API to validate and refine queries
based on actual product availability and relevance.
"""
import json
import re
from typing import Optional
import numpy as np


class ProductRetriever:
    """
    Toy product retrieval API.

    In production (Alibaba), this is a full-scale retrieval system
    (ANN + dense retrieval) over hundreds of millions of items.
    Here we use keyword matching + cosine similarity over toy embeddings.

    Paper formula (§3.2):
        P(q) = Retrieve(q, Inventory)  — top-K products for query q
        Consistency = |P(q) ∩ PurchaseIntent(u)| / K
    """

    def __init__(self, inventory: list[dict], embed_dim: int = 64):
        self.inventory = inventory
        self.embed_dim = embed_dim
        self._index = self._build_index()

    def _text_to_vec(self, text: str) -> np.ndarray:
        """Deterministic fake embedding (simulates semantic search)."""
        np.random.seed(hash(text.lower()) % (2**31))
        vec = np.random.randn(self.embed_dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _build_index(self) -> np.ndarray:
        """Build a simple embedding index over inventory."""
        vecs = [self._text_to_vec(f"{p['title']} {p['category']}") for p in self.inventory]
        return np.stack(vecs)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-K products for a query.

        Paper chain-of-retrieval: after generating a query, the agent
        calls this API to check whether retrieved products match user intent.
        """
        q_vec = self._text_to_vec(query).reshape(1, -1)
        sims = (self._index @ q_vec.T).flatten()
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_idx:
            item = dict(self.inventory[idx])
            item["retrieval_score"] = float(sims[idx])
            results.append(item)
        return results

    def compute_consistency(
        self,
        query: str,
        user_preferred_categories: list[str],
        top_k: int = 10,
    ) -> float:
        """
        Consistency score: fraction of top-K retrieved products that
        match user's purchase intent (preferred categories).

        Paper §3.3 (Consistency Reward):
            Cons(q, u) = |{p ∈ P(q) : cat(p) ∈ PrefCat(u)}| / K
        """
        retrieved = self.retrieve(query, top_k=top_k)
        pref_set = set(c.lower() for c in user_preferred_categories)
        matches = sum(1 for p in retrieved if p["category"].lower() in pref_set)
        return matches / (len(retrieved) + 1e-8)


if __name__ == "__main__":
    with open("data/inventory.json") as f:
        inventory = json.load(f)

    retriever = ProductRetriever(inventory)

    # Test retrieval
    query = "wireless noise cancelling headphones"
    results = retriever.retrieve(query, top_k=5)
    print(f"Top-5 results for '{query}':")
    for r in results:
        print(f"  [{r['retrieval_score']:.3f}] {r['title']} (cat={r['category']})")

    # Test consistency
    cons = retriever.compute_consistency(
        query,
        user_preferred_categories=["headphones", "laptop"],
        top_k=10,
    )
    print(f"\nConsistency with [headphones, laptop]: {cons:.3f}")
