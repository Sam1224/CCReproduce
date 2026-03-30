from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC without sklearn (rank-based)."""

    y_true = y_true.astype(np.int32)
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    n_pos = int(y_true_sorted.sum())
    n_neg = int(len(y_true_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # rank sum of positives (1-indexed)
    ranks = np.arange(1, len(y_true_sorted) + 1)
    pos_rank_sum = float((ranks * y_true_sorted).sum())

    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def gauc_score(groups: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Group AUC (weighted by #impressions per group)."""

    groups = groups.astype(np.int64)
    uniq = np.unique(groups)

    total = 0.0
    weight = 0.0

    for g in uniq:
        idx = groups == g
        y_g = y_true[idx]
        s_g = y_score[idx]
        n_pos = int(y_g.sum())
        n_neg = int(len(y_g) - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        a = auc_score(y_g, s_g)
        w = float(len(y_g))
        total += a * w
        weight += w

    if weight == 0:
        return 0.5
    return float(total / weight)


def train_test_split(n: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(n * (1.0 - test_ratio))
    return idx[:k], idx[k:]
