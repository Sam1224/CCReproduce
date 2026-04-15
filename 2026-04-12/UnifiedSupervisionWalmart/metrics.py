from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def dcg(relevances: List[float]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(graded_rels: List[int], k: int) -> float:
    rels = [r / 4.0 for r in graded_rels[:k]]
    dcg_v = dcg(rels)
    ideal = sorted([r / 4.0 for r in graded_rels], reverse=True)[:k]
    idcg_v = dcg(ideal)
    return 0.0 if idcg_v == 0 else dcg_v / idcg_v


def precision_at_k(graded_rels: List[int], k: int, threshold: int = 2) -> float:
    rel = sum(1 for r in graded_rels[:k] if r >= threshold)
    return rel / k


def avg_relevance_at_k(graded_rels: List[int], k: int) -> float:
    return float(np.mean([r for r in graded_rels[:k]]))
