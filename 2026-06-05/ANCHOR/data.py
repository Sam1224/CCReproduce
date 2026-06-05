from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    label: int  # 1 positive implicit feedback
    is_noise: int  # 1 noisy interaction, 0 clean
    noise_type: str


@dataclass(frozen=True)
class Dataset:
    n_users: int
    n_items: int
    interactions: list[Interaction]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_synthetic_dataset(
    *,
    n_users: int = 2000,
    n_items: int = 4000,
    clean_per_user: int = 20,
    noise_ratio: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """Synthetic implicit-feedback dataset with injected noise.

    Each user has a latent preference cluster; clean clicks come from that cluster.
    Noisy clicks are injected via multiple simulated behaviors (5 types in the paper).
    """

    seed_everything(seed)

    n_clusters = 20
    item_cluster = np.random.randint(0, n_clusters, size=n_items)
    user_cluster = np.random.randint(0, n_clusters, size=n_users)

    # Popularity distribution (for popularity/position bias)
    popularity = np.random.zipf(a=1.3, size=n_items).astype(np.float64)
    popularity = popularity / popularity.sum()

    noise_types = [
        "misclick",
        "curiosity",
        "caption_bias",
        "popularity_bias",
        "position_bias",
    ]

    interactions: list[Interaction] = []

    for u in range(n_users):
        cluster = int(user_cluster[u])
        clean_items = np.where(item_cluster == cluster)[0]
        clean_items = clean_items if len(clean_items) > 0 else np.arange(n_items)

        # Clean interactions
        for _ in range(clean_per_user):
            i = int(np.random.choice(clean_items))
            interactions.append(Interaction(user_id=u, item_id=i, label=1, is_noise=0, noise_type="clean"))

        # Noisy interactions
        n_noise = int(round(clean_per_user * noise_ratio))
        for _ in range(n_noise):
            nt = random.choice(noise_types)

            if nt == "misclick":
                i = random.randrange(n_items)
            elif nt == "curiosity":
                # sample from neighboring clusters
                other = (cluster + random.randint(1, 5)) % n_clusters
                candidates = np.where(item_cluster == other)[0]
                i = int(np.random.choice(candidates)) if len(candidates) else random.randrange(n_items)
            elif nt == "caption_bias":
                # simulate text/caption bias by choosing items sharing a token-like prefix (cluster parity)
                parity = cluster % 2
                candidates = np.where((item_cluster % 2) == parity)[0]
                i = int(np.random.choice(candidates)) if len(candidates) else random.randrange(n_items)
            elif nt == "popularity_bias":
                i = int(np.random.choice(np.arange(n_items), p=popularity))
            elif nt == "position_bias":
                # position bias: pick from top-100 frequently exposed
                top = np.argsort(-popularity)[:100]
                i = int(np.random.choice(top))
            else:
                i = random.randrange(n_items)

            interactions.append(Interaction(user_id=u, item_id=i, label=1, is_noise=1, noise_type=nt))

    random.shuffle(interactions)
    return Dataset(n_users=n_users, n_items=n_items, interactions=interactions)


def split_interactions(
    ds: Dataset, seed: int = 42
) -> tuple[list[Interaction], list[Interaction], list[Interaction]]:
    rng = random.Random(seed)
    idx = list(range(len(ds.interactions)))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = [ds.interactions[i] for i in idx[:n_train]]
    val = [ds.interactions[i] for i in idx[n_train : n_train + n_val]]
    test = [ds.interactions[i] for i in idx[n_train + n_val :]]
    return train, val, test
