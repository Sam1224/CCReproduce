from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _dcg(rels: Sequence[int], k: int) -> float:
    rels = rels[:k]
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = _dcg(relevances, k)
    idcg = _dcg(sorted(relevances, reverse=True), k)
    return 0.0 if idcg == 0 else float(dcg / idcg)


def average_precision_at_k(relevances: List[int], k: int) -> float:
    # Treat relevance > 0 as positive
    hits = 0
    s = 0.0
    for i, r in enumerate(relevances[:k], start=1):
        if r > 0:
            hits += 1
            s += hits / i
    return 0.0 if hits == 0 else float(s / hits)


@dataclass
class RetrievalMetrics:
    map_at_8: float
    ndcg_at_8: float
    auc: float


def compute_metrics_for_groups(
    model,
    tokenizer,
    device: torch.device,
    groups: List[Tuple[str, List[str]]],
    grade_map: Dict[int, int] | None = None,
    k: int = 8,
) -> RetrievalMetrics:
    """Evaluate on grouped candidates.

    groups: list of (query, ordered_products)

    grade_map maps index -> grade, default uses 3/2/1/0 for [0..3] and 0 for rest.
    """

    if grade_map is None:
        grade_map = {0: 3, 1: 2, 2: 1, 3: 0}

    map_scores: List[float] = []
    ndcg_scores: List[float] = []

    auc_labels: List[int] = []
    auc_scores: List[float] = []

    model.eval()
    with torch.no_grad():
        for query, products in groups:
            q_batch = tokenizer.encode_batch([query])
            d_batch = tokenizer.encode_batch(products)

            q_batch = type(q_batch)(
                token_ids=q_batch.token_ids.to(device),
                attn_mask=q_batch.attn_mask.to(device),
            )
            d_batch = type(d_batch)(
                token_ids=d_batch.token_ids.to(device),
                attn_mask=d_batch.attn_mask.to(device),
            )

            q_emb = model.encode(q_batch.token_ids, q_batch.attn_mask)
            d_emb = model.encode(d_batch.token_ids, d_batch.attn_mask)
            sims = (q_emb @ d_emb.t()).squeeze(0).cpu().numpy().tolist()

            # Rank by similarity
            ranked = sorted(range(len(products)), key=lambda i: sims[i], reverse=True)
            rels = [grade_map.get(i, 0) for i in ranked]

            map_scores.append(average_precision_at_k(rels, k))
            ndcg_scores.append(ndcg_at_k(rels, k))

            # AUC over pos (grade>=2) vs neg
            for i in range(len(products)):
                lbl = 1 if grade_map.get(i, 0) >= 2 else 0
                auc_labels.append(lbl)
                auc_scores.append(float(sims[i]))

    try:
        auc = float(roc_auc_score(auc_labels, auc_scores))
    except ValueError:
        auc = 0.0

    return RetrievalMetrics(
        map_at_8=float(np.mean(map_scores)),
        ndcg_at_8=float(np.mean(ndcg_scores)),
        auc=auc,
    )
