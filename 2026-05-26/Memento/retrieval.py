from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def mmr_select(
    *,
    query: np.ndarray,
    candidates: np.ndarray,
    k: int,
    lambda_relevance: float = 0.6,
) -> list[int]:
    """Maximal Marginal Relevance selection.

    Args:
        query: (d,)
        candidates: (n, d)
        k: number to select
        lambda_relevance: higher -> more relevance, lower -> more diversity

    Returns:
        indices into candidates
    """

    if k <= 0:
        return []

    n = candidates.shape[0]
    k = min(k, n)

    query = query.reshape(1, -1)
    sim_q = cosine_similarity_matrix(candidates, query).reshape(-1)  # (n,)

    selected: list[int] = []
    remaining = list(range(n))

    # Precompute candidate-candidate similarities for diversity term
    sim_cc = cosine_similarity_matrix(candidates, candidates)

    for _ in range(k):
        best_idx = None
        best_score = -1e18

        for idx in remaining:
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = float(sim_cc[idx, selected].max())

            score = lambda_relevance * float(sim_q[idx]) - (1.0 - lambda_relevance) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected
