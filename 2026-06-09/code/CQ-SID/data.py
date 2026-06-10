"""
Toy data for CQ-SID + EG-GRPO reproduction.

Simulates e-commerce search data:
- Products with category + text embeddings
- Queries with text embeddings
- Relevance labels (from ranker)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def generate_toy_data(
    n_products: int = 5000,
    n_categories: int = 50,
    n_queries: int = 10000,
    feat_dim: int = 128,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Product features (text + category)
    product_text_feats = rng.standard_normal((n_products, feat_dim)).astype(np.float32)
    product_categories = rng.integers(0, n_categories, size=n_products)

    # Query features
    query_feats = rng.standard_normal((n_queries, feat_dim)).astype(np.float32)

    # Query-product pairs with relevance scores (simulating ranker output)
    pair_query_ids = rng.integers(0, n_queries, size=n_queries * 5)
    pair_product_ids = rng.integers(0, n_products, size=n_queries * 5)

    # Relevance: dot product similarity + noise (simulating biased ranker reward)
    query_vecs = query_feats[pair_query_ids]
    product_vecs = product_text_feats[pair_product_ids]
    relevance = (query_vecs * product_vecs).sum(-1)
    relevance = (relevance - relevance.mean()) / (relevance.std() + 1e-8)
    # Binary label
    labels = (relevance > 0).astype(np.int32)

    return {
        "product_text": product_text_feats,
        "product_cats": product_categories,
        "query_feats": query_feats,
        "pair_queries": pair_query_ids,
        "pair_products": pair_product_ids,
        "labels": labels,
        "n_products": n_products,
        "n_categories": n_categories,
        "n_queries": n_queries,
        "feat_dim": feat_dim,
    }


class EcomSearchDataset(Dataset):
    def __init__(self, data, split="train", train_ratio=0.8):
        n = len(data["labels"])
        split_idx = int(n * train_ratio)
        idx = slice(None, split_idx) if split == "train" else slice(split_idx, None)

        self.query_ids = torch.tensor(data["pair_queries"][idx], dtype=torch.long)
        self.product_ids = torch.tensor(data["pair_products"][idx], dtype=torch.long)
        self.labels = torch.tensor(data["labels"][idx], dtype=torch.float32)

        self.query_feats = torch.tensor(data["query_feats"], dtype=torch.float32)
        self.product_text = torch.tensor(data["product_text"], dtype=torch.float32)
        self.product_cats = torch.tensor(data["product_cats"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        qid = self.query_ids[idx]
        pid = self.product_ids[idx]
        return {
            "query_id": qid,
            "product_id": pid,
            "query_feat": self.query_feats[qid],
            "product_feat": self.product_text[pid],
            "product_cat": self.product_cats[pid],
            "label": self.labels[idx],
        }


def get_dataloaders(batch_size=256, **kwargs):
    data = generate_toy_data(**kwargs)
    train_ds = EcomSearchDataset(data, "train")
    val_ds = EcomSearchDataset(data, "val")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        data,
    )
