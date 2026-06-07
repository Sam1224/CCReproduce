"""
Product retrieval engine for QueryAgent-R1.
Maps query token IDs → query embedding → top-K product embeddings.

In production: backed by ANN index (Faiss / HNSW) over million-scale product catalog.
Here: toy in-memory index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QueryEncoder(nn.Module):
    """Encodes query token IDs into a dense query vector."""

    def __init__(self, vocab_size: int = 10_000, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        # query_ids: (B, L)
        emb = self.token_embed(query_ids).mean(dim=1)  # (B, D) mean pool
        return F.normalize(self.pool_proj(emb), dim=-1)


class ProductCatalog:
    """
    Toy product catalog.
    Stores product embeddings and supports ANN-style retrieval.
    """

    def __init__(self, num_products: int = 5_000, embed_dim: int = 64):
        # Random product embeddings (in production: real product representations)
        self.embeddings = F.normalize(
            torch.randn(num_products, embed_dim), dim=-1
        )
        self.num_products = num_products
        self.embed_dim = embed_dim

    def retrieve(self, query_emb: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (product_embeddings, product_ids) for top-k products per query.
        query_emb: (B, D) normalized query embeddings
        """
        sim = query_emb @ self.embeddings.T  # (B, P)
        top_scores, top_ids = sim.topk(top_k, dim=-1)  # (B, K)
        top_embs = self.embeddings[top_ids]  # (B, K, D)
        return top_embs, top_ids


class ProductRetriever(nn.Module):
    """
    Full retrieval component: encodes query → retrieves products.
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embed_dim: int = 64,
        num_products: int = 5_000,
        top_k: int = 10,
    ):
        super().__init__()
        self.query_encoder = QueryEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.catalog = ProductCatalog(num_products=num_products, embed_dim=embed_dim)
        self.top_k = top_k

    def forward(self, query_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_emb = self.query_encoder(query_ids)
        product_embs, product_ids = self.catalog.retrieve(query_emb, self.top_k)
        return product_embs, product_ids, query_emb
