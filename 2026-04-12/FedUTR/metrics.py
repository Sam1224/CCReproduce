from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np


def _dcg(relevances: List[float]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranked_items: List[int], ground_truth_item: int, k: int) -> float:
    topk = ranked_items[:k]
    rels = [1.0 if item == ground_truth_item else 0.0 for item in topk]
    dcg = _dcg(rels)
    idcg = _dcg([1.0] + [0.0] * (k - 1))
    return 0.0 if idcg == 0 else dcg / idcg


def hit_rate_at_k(ranked_items: List[int], ground_truth_item: int, k: int) -> float:
    return 1.0 if ground_truth_item in ranked_items[:k] else 0.0


def evaluate_ranking(
    scores: np.ndarray,
    ground_truth_item: int,
    k: int,
) -> Tuple[float, float]:
    ranked = list(np.argsort(-scores))
    return hit_rate_at_k(ranked, ground_truth_item, k), ndcg_at_k(ranked, ground_truth_item, k)
