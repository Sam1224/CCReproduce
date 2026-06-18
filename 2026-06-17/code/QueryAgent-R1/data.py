"""
QueryAgent-R1 — Data utilities

Simulates:
  - User query history (past search queries)
  - Product catalog with text embeddings
  - User purchase labels (CVR signal)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


def generate_toy_ecommerce_data(
    n_users: int = 2000,
    n_queries: int = 10000,
    n_products: int = 50000,
    embed_dim: int = 128,
    seq_len: int = 10,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)

    # Query embeddings (simulating text embeddings of search queries)
    query_embs = rng.standard_normal((n_queries, embed_dim)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8

    # Product embeddings
    product_embs = rng.standard_normal((n_products, embed_dim)).astype(np.float32)
    product_embs /= np.linalg.norm(product_embs, axis=1, keepdims=True) + 1e-8

    # User latent preferences
    user_prefs = rng.standard_normal((n_users, embed_dim)).astype(np.float32)

    # Query history per user
    query_seqs = []
    for u in range(n_users):
        sim = query_embs @ user_prefs[u]
        probs = np.exp(sim / 0.5); probs /= probs.sum()
        seq = rng.choice(n_queries, size=seq_len, p=probs, replace=True)
        query_seqs.append(seq)
    query_seqs = np.array(query_seqs)

    # Next query to recommend (target)
    next_query = []
    for u in range(n_users):
        sim = query_embs @ user_prefs[u]
        probs = np.exp(sim / 0.5); probs /= probs.sum()
        next_query.append(rng.choice(n_queries, p=probs))
    next_query = np.array(next_query)

    # Products retrieved by each query (top-5 neighbors by embedding)
    # Pre-compute for efficiency (in real system this is a retrieval engine)
    query_retrieved_products = []
    for q in range(n_queries):
        sims = product_embs @ query_embs[q]
        top5 = np.argsort(sims)[-5:][::-1]
        query_retrieved_products.append(top5)
    query_retrieved_products = np.array(query_retrieved_products)

    # CVR signal: whether retrieved products match user preference
    # (proxy for conversion: product_emb · user_pref)
    cvr_scores = product_embs @ user_prefs.T  # (n_products, n_users)

    return {
        "n_users": n_users,
        "n_queries": n_queries,
        "n_products": n_products,
        "embed_dim": embed_dim,
        "query_embs": query_embs,
        "product_embs": product_embs,
        "user_prefs": user_prefs,
        "query_seqs": query_seqs,
        "next_query": next_query,
        "query_retrieved_products": query_retrieved_products,
        "cvr_scores": cvr_scores,
    }


class QueryRecDataset(Dataset):
    def __init__(self, data: Dict, user_ids: np.ndarray):
        self.user_ids = user_ids
        self.query_seqs = torch.LongTensor(data["query_seqs"][user_ids])
        self.next_query = torch.LongTensor(data["next_query"][user_ids])
        # Retrieved products for the next query
        self.retrieved_prods = torch.LongTensor(
            data["query_retrieved_products"][data["next_query"][user_ids]]
        )
        # CVR signal: mean CVR of retrieved products for each user
        cvr = data["cvr_scores"]  # (n_products, n_users)
        ret_prods = data["query_retrieved_products"][data["next_query"][user_ids]]
        self.cvr_signal = torch.FloatTensor(
            np.array([cvr[ret_prods[i], user_ids[i]].mean() for i in range(len(user_ids))])
        )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "query_seq": self.query_seqs[idx],
            "next_query": self.next_query[idx],
            "retrieved_prods": self.retrieved_prods[idx],
            "cvr_signal": self.cvr_signal[idx],
        }


def build_dataloaders(data: Dict, train_ratio: float = 0.8, batch_size: int = 128):
    n = data["n_users"]
    split = int(n * train_ratio)
    ids = np.arange(n)

    def make_loader(user_ids, shuffle):
        ds = QueryRecDataset(data, user_ids)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return make_loader(ids[:split], True), make_loader(ids[split:], False)


if __name__ == "__main__":
    data = generate_toy_ecommerce_data()
    train_loader, val_loader = build_dataloaders(data)
    batch = next(iter(train_loader))
    print("Query seq shape:", batch["query_seq"].shape)
    print("Retrieved prods shape:", batch["retrieved_prods"].shape)
    print("CVR signal shape:", batch["cvr_signal"].shape)
    print("Data OK.")
