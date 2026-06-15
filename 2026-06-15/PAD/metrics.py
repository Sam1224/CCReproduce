from __future__ import annotations

import numpy as np


def recall_ndcg_at_k(scores: np.ndarray, target: int, k: int) -> tuple[float, float]:
    # scores: [n_items]
    rank = int(np.argsort(-scores).tolist().index(target)) + 1
    hit = 1.0 if rank <= k else 0.0
    ndcg = 0.0
    if hit:
        ndcg = 1.0 / np.log2(rank + 1)
    return hit, ndcg


def eval_ranking(
    user_scores: np.ndarray,  # [n_users, n_items]
    targets: np.ndarray,  # [n_users]
    ks: tuple[int, ...] = (20, 50),
) -> dict:
    out: dict[str, float] = {}
    for k in ks:
        rec, ndcg = [], []
        for u in range(user_scores.shape[0]):
            r, n = recall_ndcg_at_k(user_scores[u], int(targets[u]), k)
            rec.append(r)
            ndcg.append(n)
        out[f"recall@{k}"] = float(np.mean(rec))
        out[f"ndcg@{k}"] = float(np.mean(ndcg))
    return out


def coverage_at_k(topk_items: np.ndarray, n_items: int) -> float:
    # topk_items: [n_users, k]
    uniq = np.unique(topk_items.reshape(-1))
    return float(len(uniq)) / float(n_items)


def gini_diversity(topk_items: np.ndarray, n_items: int) -> float:
    # exposure counts
    counts = np.zeros(n_items, dtype=np.int64)
    flat = topk_items.reshape(-1)
    for i in flat:
        counts[int(i)] += 1

    if counts.sum() == 0:
        return 0.0

    x = np.sort(counts.astype(np.float64))
    n = x.size
    cum = np.cumsum(x)
    gini = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    # higher is better
    return float(1.0 - gini)
