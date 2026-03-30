from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np


@dataclass
class Batch:
    user: np.ndarray  # (B,)
    item: np.ndarray  # (B,)
    label: np.ndarray  # (B,) float32


def make_toy_implicit_data(
    n_users: int,
    n_items: int,
    n_pos: int,
    seed: int = 0,
) -> List[Tuple[int, int, float]]:
    """Generate a toy implicit feedback dataset.

    We sample positive interactions, and we will sample negatives on the fly.
    """
    rng = np.random.default_rng(seed)
    users = rng.integers(0, n_users, size=n_pos)
    items = rng.integers(0, n_items, size=n_pos)
    data = [(int(u), int(i), 1.0) for u, i in zip(users, items)]
    return data


def split_train_val(
    data: List[Tuple[int, int, float]],
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))
    train = [data[i] for i in idx[:cut]]
    val = [data[i] for i in idx[cut:]]
    return train, val


def sample_batch(
    data: List[Tuple[int, int, float]],
    n_users: int,
    n_items: int,
    batch_size: int,
    neg_ratio: int = 1,
    seed: int = 0,
) -> Batch:
    """Sample a mixed batch with negative sampling."""
    rng = np.random.default_rng(seed)
    pos_idx = rng.integers(0, len(data), size=batch_size)
    pos = [data[i] for i in pos_idx]

    user: List[int] = []
    item: List[int] = []
    label: List[float] = []

    for u, it, y in pos:
        user.append(u)
        item.append(it)
        label.append(y)
        for _ in range(neg_ratio):
            user.append(u)
            item.append(int(rng.integers(0, n_items)))
            label.append(0.0)

    return Batch(
        user=np.asarray(user, dtype=np.int64),
        item=np.asarray(item, dtype=np.int64),
        label=np.asarray(label, dtype=np.float32),
    )


def stream_batches(
    data: List[Tuple[int, int, float]],
    n_users: int,
    n_items: int,
    batch_size: int,
    neg_ratio: int,
    seed: int = 0,
) -> Iterator[Batch]:
    """Infinite stream of batches."""
    step = 0
    while True:
        yield sample_batch(
            data=data,
            n_users=n_users,
            n_items=n_items,
            batch_size=batch_size,
            neg_ratio=neg_ratio,
            seed=seed + step,
        )
        step += 1
