from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from data import Product, Query, relevance_grade


def _tok(s: str) -> List[str]:
    return [t for t in s.lower().replace("-", " ").split() if t]


def lexical_score(q: Query, p: Product) -> float:
    qt = set(_tok(q.text))
    pt = set(_tok(p.title))
    if not qt or not pt:
        return 0.0
    inter = len(qt & pt)
    union = len(qt | pt)
    return inter / union


def dense_baseline_score(q: Query, p: Product, proj: Dict[str, np.ndarray]) -> float:
    """A deterministic dense-like baseline using random projections on tokens.

    This is only a *proxy* for a learned dense retriever. It is used to create
    multi-channel disagreement signals for structured mining.
    """

    qv = np.zeros_like(next(iter(proj.values())))
    dv = np.zeros_like(qv)
    for t in _tok(q.text):
        if t in proj:
            qv += proj[t]
    for t in _tok(p.title):
        if t in proj:
            dv += proj[t]
    qn = np.linalg.norm(qv) + 1e-9
    dn = np.linalg.norm(dv) + 1e-9
    return float(np.dot(qv, dv) / (qn * dn))


def build_token_projection(queries: Sequence[Query], products: Sequence[Product], dim: int = 48, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = set()
    for q in queries:
        tokens.update(_tok(q.text))
    for p in products:
        tokens.update(_tok(p.title))
    proj: Dict[str, np.ndarray] = {}
    for t in sorted(tokens):
        v = rng.normal(0, 1.0, size=(dim,)).astype(np.float32)
        proj[t] = v / (np.linalg.norm(v) + 1e-9)
    return proj


def topk_indices(scores: Sequence[float], k: int) -> List[int]:
    if k >= len(scores):
        return list(np.argsort(-np.array(scores)))
    return list(np.argpartition(-np.array(scores), k)[:k])


def mine_pairs(
    queries: Sequence[Query],
    products: Sequence[Product],
    topk: int = 20,
    per_query_pairs: int = 24,
    seed: int = 0,
):
    """Structured mining from multi-channel disagreement.

    Returns a list of pair dicts:
      {qid, pid, grade, bucket}

    Buckets are inspired by the paper's idea of using disagreements to mine
    easy/hard positives and hard negatives.
    """

    rng = random.Random(seed)
    proj = build_token_projection(queries, products, seed=seed)

    by_pid = {p.pid: p for p in products}
    pairs = []
    bucket_counts = defaultdict(int)

    for q in queries:
        lex_scores = [lexical_score(q, p) for p in products]
        dense_scores = [dense_baseline_score(q, p, proj) for p in products]

        lex_top = set(topk_indices(lex_scores, k=topk))
        dense_top = set(topk_indices(dense_scores, k=topk))
        union = lex_top | dense_top

        # Candidate pool from disagreement signals
        candidates = list(union)

        # Add a few random negatives outside union to keep easy negatives.
        outside = [i for i in range(len(products)) if i not in union]
        rng.shuffle(outside)
        candidates.extend(outside[: max(8, topk // 2)])

        # De-duplicate
        candidates = list(dict.fromkeys(candidates))

        local = []
        for idx in candidates:
            p = products[idx]
            grade = relevance_grade(q, p)

            in_lex = idx in lex_top
            in_dense = idx in dense_top

            if grade >= 3 and in_lex and in_dense:
                bucket = "easy_pos"
            elif grade >= 3 and in_lex and not in_dense:
                bucket = "hard_pos"
            elif grade == 0 and in_dense and not in_lex:
                bucket = "hard_neg"
            elif grade == 0 and (not in_lex) and (not in_dense):
                bucket = "easy_neg"
            else:
                bucket = "medium"

            local.append({"qid": q.qid, "pid": p.pid, "grade": int(grade), "bucket": bucket})

        # Keep a balanced subset per query.
        by_bucket: Dict[str, List[dict]] = defaultdict(list)
        for it in local:
            by_bucket[it["bucket"]].append(it)

        # target mix: emphasize positives and hard negatives
        targets = {
            "easy_pos": per_query_pairs // 6,
            "hard_pos": per_query_pairs // 6,
            "hard_neg": per_query_pairs // 4,
            "medium": per_query_pairs // 4,
            "easy_neg": per_query_pairs // 4,
        }

        picked = []
        for b, n in targets.items():
            pool = by_bucket.get(b, [])
            rng.shuffle(pool)
            picked.extend(pool[:n])

        # ensure at least one positive exists (for later curriculum stages)
        if not any(it["grade"] >= 3 for it in picked):
            pos_pool = [it for it in local if it["grade"] >= 3]
            if pos_pool:
                picked.append(rng.choice(pos_pool))

        for it in picked:
            pairs.append(it)
            bucket_counts[it["bucket"]] += 1

    return pairs, dict(bucket_counts)
