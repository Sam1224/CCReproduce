from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)


@dataclass(frozen=True)
class ToyCFConfig:
    num_users: int = 200
    num_items: int = 300
    interactions_per_user: int = 30
    seed: int = 42


@dataclass(frozen=True)
class ToyCFSplit:
    # Each list element is (u, i) with u in [0, U), i in [0, I)
    train_edges: list[tuple[int, int]]
    val_edges: list[tuple[int, int]]
    test_edges: list[tuple[int, int]]

    # For evaluation: per-user ground truth
    val_pos_by_user: dict[int, set[int]]
    test_pos_by_user: dict[int, set[int]]

    # For masking: training positives
    train_pos_by_user: dict[int, set[int]]


def build_toy_cf_split(cfg: ToyCFConfig) -> ToyCFSplit:
    """Build a small implicit-feedback dataset with per-user train/val/test split."""

    set_seed(cfg.seed)

    train_edges: list[tuple[int, int]] = []
    val_edges: list[tuple[int, int]] = []
    test_edges: list[tuple[int, int]] = []

    train_pos_by_user: dict[int, set[int]] = {}
    val_pos_by_user: dict[int, set[int]] = {}
    test_pos_by_user: dict[int, set[int]] = {}

    for u in range(cfg.num_users):
        items = np.random.choice(cfg.num_items, size=cfg.interactions_per_user, replace=False)
        items = items.tolist()

        # 1 val, 1 test, rest train
        val_i = items[-2]
        test_i = items[-1]
        train_is = items[:-2]

        train_pos_by_user[u] = set(train_is)
        val_pos_by_user[u] = {val_i}
        test_pos_by_user[u] = {test_i}

        for i in train_is:
            train_edges.append((u, i))
        val_edges.append((u, val_i))
        test_edges.append((u, test_i))

    return ToyCFSplit(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        train_pos_by_user=train_pos_by_user,
        val_pos_by_user=val_pos_by_user,
        test_pos_by_user=test_pos_by_user,
    )
