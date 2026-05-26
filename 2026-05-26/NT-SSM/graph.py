from __future__ import annotations

import numpy as np


def build_adjacency(*, num_users: int, num_items: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build (U+I)x(U+I) adjacency for bipartite graph."""

    n = num_users + num_items
    a = np.zeros((n, n), dtype=np.float32)

    for u, i in edges:
        iu = u
        ii = num_users + i
        a[iu, ii] = 1.0
        a[ii, iu] = 1.0

    return a


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""

    deg = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.clip(deg, a_min=1.0, a_max=None))
    d_inv_sqrt = np.diag(inv_sqrt.astype(np.float32))
    return d_inv_sqrt @ a @ d_inv_sqrt


def compute_s_tilde(*, a_norm: np.ndarray, num_layers: int) -> np.ndarray:
    """Compute \tilde{S} = (1/(L+1)) * sum_{l=0..L} (A_norm^l)."""

    n = a_norm.shape[0]
    out = np.eye(n, dtype=np.float32)
    cur = np.eye(n, dtype=np.float32)

    for _ in range(num_layers):
        cur = cur @ a_norm
        out = out + cur

    out = out / float(num_layers + 1)
    return out.astype(np.float32)
