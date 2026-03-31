from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Pair:
    q: torch.Tensor  # (Lq,)
    ad_title: torch.Tensor  # (Lt,)
    ad_img: torch.Tensor  # (d,)
    y: int
    ad_id: int


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_pairs(
    n_ads: int = 4000,
    n_pairs: int = 60000,
    vocab: int = 5000,
    d_img: int = 64,
    pos_rate: float = 1e-3,
    seed: int = 0,
) -> Tuple[List[Pair], torch.Tensor]:
    """Generate heavily imbalanced (query, ad) pairs.

    We simulate ad embeddings + textual titles. Offensive pairing depends on
    query intent interacting with ad embedding.

    Returns:
      pairs: examples
      ad_embs: (n_ads, d_img)
    """

    rng = _seed(seed)
    ad_embs = rng.normal(size=(n_ads, d_img)).astype(np.float32)
    # introduce a few "offensive" ad clusters
    n_clusters = 8
    cluster_centers = rng.normal(size=(n_clusters, d_img)).astype(np.float32)
    cluster_ids = rng.integers(0, n_clusters, size=(n_ads,))
    ad_embs += cluster_centers[cluster_ids] * 0.7

    pairs: List[Pair] = []
    for _ in range(n_pairs):
        ad_id = int(rng.integers(0, n_ads))
        q_len = int(rng.integers(2, 8))
        t_len = int(rng.integers(4, 12))

        q = rng.integers(0, vocab, size=(q_len,), dtype=np.int64)
        title = rng.integers(0, vocab, size=(t_len,), dtype=np.int64)
        img = ad_embs[ad_id]

        # Pair offensiveness: rare, but correlated with cluster id and a subset of query tokens.
        intent = int(q[0] % n_clusters)
        y = 1 if (intent == int(cluster_ids[ad_id]) and rng.random() < 0.18) else 0
        # enforce global scarcity
        if rng.random() > pos_rate / max(pos_rate, 0.18):
            y = 0

        pairs.append(Pair(q=torch.tensor(q), ad_title=torch.tensor(title), ad_img=torch.tensor(img), y=y, ad_id=ad_id))

    return pairs, torch.tensor(ad_embs)


def split(items: List[Pair], frac: float = 0.9) -> Tuple[List[Pair], List[Pair]]:
    n = int(len(items) * frac)
    return items[:n], items[n:]
