"""
Product Retrieval Module for QueryAgent-R1
Paper: arxiv 2606.05671

Retrieval is integrated into the RL training loop — the agent sees retrieved
products as feedback to validate and refine its generated queries.

Supports:
  - Dense retrieval via pre-computed FAISS index
  - BM25 lexical retrieval (fallback / ensemble)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class ProductRetriever:
    """
    Product retrieval system used to validate agent-generated queries.
    
    In production: backed by platform's multi-billion product catalog.
    Toy version: in-memory catalog with random embeddings.
    
    From paper Section 3.2:
        The agent calls retrieval after generating a candidate query q̂.
        Retrieved products P(q̂) are compared to user's target products
        to compute the retrieval consistency reward.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        catalog_size: int = 10000,
    ):
        self.embed_dim = embed_dim
        # Toy catalog: product_id → (embedding, metadata)
        self.catalog_embeddings = F.normalize(
            torch.randn(catalog_size, embed_dim), dim=-1
        )
        self.catalog_meta = [
            {
                "product_id": f"pid_{i:06d}",
                "title": f"Product {i}",
                "category": f"cat_{i % 20}",
                "price": round(10 + (i % 990) * 0.1, 2),
            }
            for i in range(catalog_size)
        ]

    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode query text to dense embedding.
        Production: use a fine-tuned query encoder (e.g., bilingual E5/BGE).
        Toy: hash-based deterministic embedding.
        """
        # Deterministic toy embedding based on query hash
        seed = hash(query) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        emb = F.normalize(torch.randn(self.embed_dim, generator=gen), dim=-1)
        return emb

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Retrieve top-k products for a given query.
        
        Returns list of {product_id, title, category, score} dicts.
        """
        query_emb = self.encode_query(query)  # [embed_dim]
        scores = (query_emb @ self.catalog_embeddings.T)  # [catalog_size]
        top_indices = scores.topk(top_k).indices

        results = []
        for idx in top_indices:
            meta = self.catalog_meta[idx.item()].copy()
            meta["retrieval_score"] = scores[idx].item()
            results.append(meta)
        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[Dict]]:
        """Batch retrieval for RL training efficiency."""
        return [self.retrieve(q, top_k) for q in queries]


class UserHistoryEncoder:
    """
    Encode user interaction history into an interest graph for memory abstraction.
    
    From paper Section 3.3 (Memory Abstraction Module):
        Interest graph G_u = {(item_i, category_i, weight_i)} 
        extracted from user's long-term click/purchase history.
    
    Compressed representation: [behavior_type, category, weight] tuples
    → dense interest embedding via graph aggregation.
    """

    def __init__(self, embed_dim: int = 256, num_categories: int = 20):
        self.embed_dim = embed_dim
        # Category embeddings (learnable in training)
        self.cat_embeddings = nn.Embedding(num_categories, embed_dim) if False else \
            torch.randn(num_categories, embed_dim)
        self.cat_embeddings = F.normalize(
            torch.tensor(self.cat_embeddings) if not isinstance(
                self.cat_embeddings, torch.Tensor
            ) else self.cat_embeddings,
            dim=-1
        )

    def encode(self, history: List[Dict]) -> torch.Tensor:
        """
        Encode user history to interest graph embedding.
        
        Args:
            history: list of {product_id, category_idx, weight, timestamp} dicts
        
        Returns:
            interest_embedding: [embed_dim] weighted aggregation
        """
        if not history:
            return torch.zeros(self.embed_dim)

        # Weighted sum of category embeddings
        weights = torch.tensor([h.get("weight", 1.0) for h in history])
        weights = weights / weights.sum()

        cat_indices = torch.tensor([
            h.get("category_idx", 0) % len(self.cat_embeddings)
            for h in history
        ])
        cat_embs = self.cat_embeddings[cat_indices]  # [N, embed_dim]
        interest_emb = (cat_embs * weights.unsqueeze(1)).sum(dim=0)  # [embed_dim]
        return F.normalize(interest_emb, dim=-1)
