from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ToyBatch:
    cat: np.ndarray  # (B,)
    x: np.ndarray  # (B, d)
    y: np.ndarray  # (B,) float32


def make_toy_ranking_data(n: int = 5000, d: int = 8, n_cat: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, n_cat, size=n)
    x = rng.standard_normal((n, d)).astype(np.float32)

    # hidden ground-truth
    w = rng.standard_normal((d,)).astype(np.float32)
    cat_bias = rng.standard_normal((n_cat,)).astype(np.float32) * 0.2
    logit = (x @ w) + cat_bias[cat]
    y = (logit > 0).astype(np.float32)

    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr = idx[:cut]
    va = idx[cut:]

    train = (cat[tr], x[tr], y[tr])
    val = (cat[va], x[va], y[va])
    return train, val


def iter_batches(cat, x, y, batch_size: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    while True:
        idx = rng.integers(0, n, size=batch_size)
        yield ToyBatch(cat=cat[idx], x=x[idx], y=y[idx])
