from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Batch:
    x_text: np.ndarray  # (B, d)
    x_img: np.ndarray  # (B, d)
    y: np.ndarray  # (B,) int64
    domain: np.ndarray  # (B,) int64


def make_dataset(
    n: int = 20000,
    d: int = 16,
    n_domains: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic multimodal classification with domain-specific shifts.

    Each domain makes *different feature blocks* predictive. This mimics a setting
    where different experts are better for different input clusters.
    """

    rng = np.random.default_rng(seed)
    domain = rng.integers(0, n_domains, size=n, dtype=np.int64)

    x_text = rng.normal(size=(n, d)).astype(np.float32)
    x_img = rng.normal(size=(n, d)).astype(np.float32)

    # Domain-specific ground-truth hyperplanes.
    # Domain k emphasizes different slices.
    w_text = rng.normal(size=(n_domains, d)).astype(np.float32)
    w_img = rng.normal(size=(n_domains, d)).astype(np.float32)

    for k in range(n_domains):
        # encourage separability by boosting a small block
        start = (k * (d // n_domains)) % d
        end = min(d, start + (d // n_domains))
        w_text[k, start:end] *= 3.0
        w_img[k, start:end] *= 3.0

    logits = np.zeros(n, dtype=np.float32)
    for k in range(n_domains):
        idx = domain == k
        if not np.any(idx):
            continue
        logits[idx] = (x_text[idx] * w_text[k]).sum(axis=1) + (x_img[idx] * w_img[k]).sum(axis=1)

    # Add noise; label is sign(logit)
    logits = logits + 0.5 * rng.normal(size=n).astype(np.float32)
    y = (logits > 0).astype(np.int64)

    return x_text, x_img, y, domain


def split_train_val(
    x_text: np.ndarray,
    x_img: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))

    tr = idx[:cut]
    va = idx[cut:]

    return (
        (x_text[tr], x_img[tr], y[tr], domain[tr]),
        (x_text[va], x_img[va], y[va], domain[va]),
    )


def iter_batches(
    x_text: np.ndarray,
    x_img: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    batch_size: int = 256,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = len(y)
    step = 0
    while True:
        idx = rng.integers(0, n, size=batch_size)
        yield Batch(
            x_text=x_text[idx],
            x_img=x_img[idx],
            y=y[idx],
            domain=domain[idx],
        )
        step += 1
