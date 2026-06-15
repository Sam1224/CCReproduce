from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ToyConfig:
    n_users: int = 2000
    n_items: int = 4000
    latent_dim: int = 32
    train_pos_per_user: int = 8
    test_pos_per_user: int = 1
    neg_per_pos: int = 4

    # popularity prior (Zipf)
    zipf_a: float = 1.05

    # noise: add false-positive interactions on popular head
    head_frac: float = 0.05
    noise_pos_per_user: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _zipf_probs(n: int, a: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, a)
    return w / w.sum()


def generate_dataset(cfg: ToyConfig, seed: int) -> Dict:
    rng = np.random.default_rng(seed)

    # ground-truth embeddings
    u_true = rng.normal(size=(cfg.n_users, cfg.latent_dim)).astype(np.float32)
    i_true = rng.normal(size=(cfg.n_items, cfg.latent_dim)).astype(np.float32)

    pop_prior = _zipf_probs(cfg.n_items, cfg.zipf_a)
    head_n = max(1, int(math.floor(cfg.n_items * cfg.head_frac)))
    head_items = np.arange(head_n, dtype=np.int64)

    # clean interactions
    train_pos: List[Tuple[int, int]] = []
    test_pos: List[Tuple[int, int]] = []

    for u in range(cfg.n_users):
        logits = (i_true @ u_true[u]).astype(np.float64)
        logits = logits + 2.0 * np.log(pop_prior + 1e-12)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

        chosen = rng.choice(cfg.n_items, size=cfg.train_pos_per_user + cfg.test_pos_per_user, replace=False, p=probs)
        train = chosen[: cfg.train_pos_per_user]
        test = chosen[cfg.train_pos_per_user :]

        for it in train:
            train_pos.append((u, int(it)))
        for it in test:
            test_pos.append((u, int(it)))

    # observed positives = clean train + extra noisy head interactions
    observed_pos = set(train_pos)
    for u in range(cfg.n_users):
        noisy = rng.choice(head_items, size=cfg.noise_pos_per_user, replace=False)
        for it in noisy:
            observed_pos.add((u, int(it)))

    # popularity computed from observed positives (collaborative signal)
    item_pop = np.zeros(cfg.n_items, dtype=np.int64)
    for _, it in observed_pos:
        item_pop[it] += 1

    return {
        "train_pos_clean": train_pos,
        "train_pos_observed": sorted(observed_pos),
        "test_pos_clean": test_pos,
        "item_pop": item_pop,
    }


def build_bce_samples(
    cfg: ToyConfig, data: Dict, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1)

    positives = data["train_pos_observed"]
    pos_u = np.array([u for u, _ in positives], dtype=np.int64)
    pos_i = np.array([i for _, i in positives], dtype=np.int64)

    # negative sampling (uniform for simplicity)
    neg_u, neg_i = [], []
    pos_set = set(positives)
    for u, _ in positives:
        cnt = 0
        while cnt < cfg.neg_per_pos:
            j = int(rng.integers(0, cfg.n_items))
            if (u, j) in pos_set:
                continue
            neg_u.append(u)
            neg_i.append(j)
            cnt += 1

    u = np.concatenate([pos_u, np.array(neg_u, dtype=np.int64)])
    i = np.concatenate([pos_i, np.array(neg_i, dtype=np.int64)])
    y = np.concatenate([
        np.ones_like(pos_u, dtype=np.float32),
        np.zeros(len(neg_u), dtype=np.float32),
    ])

    # shuffle
    perm = rng.permutation(len(u))
    return u[perm], i[perm], y[perm]


def build_test_targets(cfg: ToyConfig, data: Dict) -> np.ndarray:
    # one target per user
    tgt = np.full(cfg.n_users, -1, dtype=np.int64)
    for u, it in data["test_pos_clean"]:
        tgt[u] = it
    return tgt
