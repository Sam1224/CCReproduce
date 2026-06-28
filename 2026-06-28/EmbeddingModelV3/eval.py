from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from data import Product, Query, relevance_grade


def ndcg_at_k(grades: Sequence[int], k: int = 10) -> float:
    k = min(k, len(grades))
    gains = np.array([(2 ** g - 1) for g in grades[:k]], dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())

    ideal = sorted(grades, reverse=True)
    ideal_g = np.array([(2 ** g - 1) for g in ideal[:k]], dtype=np.float32)
    idcg = float((ideal_g * discounts).sum())
    return 0.0 if idcg <= 1e-9 else dcg / idcg


def recall_at_k(grades: Sequence[int], k: int = 10, positive_grade: int = 2) -> float:
    k = min(k, len(grades))
    return float(any(g >= positive_grade for g in grades[:k]))


@torch.no_grad()
def evaluate_retrieval(
    model,
    vocab: Dict[str, int],
    queries: Sequence[Query],
    products: Sequence[Product],
    device: str = "cpu",
    k: int = 10,
) -> Dict[str, float]:
    from data import encode
    from model import make_mask, pad_2d

    model.eval()
    model.to(device)

    q_ids = [encode(q.text, vocab) for q in queries]
    d_ids = [encode(p.title, vocab) for p in products]

    q_pad = pad_2d(q_ids, pad_id=0).to(device)
    d_pad = pad_2d(d_ids, pad_id=0).to(device)
    q_mask = make_mask(q_pad).to(device)
    d_mask = make_mask(d_pad).to(device)

    q_emb = model.encode_query(q_pad, q_mask)
    d_emb = model.encode_doc(d_pad, d_mask)

    sim = q_emb @ d_emb.T  # [Q, D]
    sim = sim.cpu().numpy()

    ndcgs = []
    recalls = []
    tail_ndcgs = []
    tail_recalls = []

    # define "tail" as rare brands (in query intents)
    brand_freq: Dict[str, int] = {}
    for q in queries:
        b = q.intent.get("brand")
        if b:
            brand_freq[b] = brand_freq.get(b, 0) + 1
    rare_brands = {b for b, c in brand_freq.items() if c <= max(1, int(len(queries) * 0.05))}

    for i, q in enumerate(queries):
        order = np.argsort(-sim[i])
        ranked = [relevance_grade(q, products[j]) for j in order]
        nd = ndcg_at_k(ranked, k=k)
        rc = recall_at_k(ranked, k=k)
        ndcgs.append(nd)
        recalls.append(rc)

        if q.intent.get("brand") in rare_brands:
            tail_ndcgs.append(nd)
            tail_recalls.append(rc)

    out = {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"recall@{k}": float(np.mean(recalls)),
        f"tail_ndcg@{k}": float(np.mean(tail_ndcgs) if tail_ndcgs else 0.0),
        f"tail_recall@{k}": float(np.mean(tail_recalls) if tail_recalls else 0.0),
    }
    return out
