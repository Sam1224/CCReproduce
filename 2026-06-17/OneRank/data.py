from __future__ import annotations

"""Toy data generator for the OneRank reproduction.

Each sample mimics a single (user, request) example fed to an industrial
multi-task ranker:

  * an interaction-history sequence of item ids,
  * a (shorter) preference-anchor sequence of item ids,
  * a set of N candidate items to be ranked, and
  * 3 binary task labels per candidate: click / cart / order, with a realistic
    nested dependency  order ⊂ cart ⊂ click.

NOTE (vs. the real paper): OneRank is trained on Shopee proprietary logs with
real candidate embeddings produced by upstream retrieval. Here we synthesize a
ground-truth low-rank user/item factor model so that the labels are *learnable*
from item ids + history, and we return candidate *ids* (the model embeds them
into "candidate embeddings" inside its structured tokenizer).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


# Fixed seed for the ground-truth factor model so that train / test splits share
# the same underlying "world" (item semantics + task thresholds).
_FACTOR_SEED = 12345


@dataclass(frozen=True)
class ToyConfig:
    n_items: int = 200
    factor_dim: int = 8
    hist_len: int = 8          # interaction-history length H
    anchor_len: int = 4        # preference-anchor length A
    n_candidates: int = 6      # N candidates per request
    # label positive rates (top-q by relevance); enforces order ⊂ cart ⊂ click
    click_q: float = 0.50
    cart_q: float = 0.25
    order_q: float = 0.10
    # sampling temperature for history/anchor (higher = more peaked on prefs)
    hist_temp: float = 10.0


def _factor_matrix(cfg: ToyConfig) -> np.ndarray:
    rng = np.random.default_rng(_FACTOR_SEED)
    F = rng.normal(size=(cfg.n_items, cfg.factor_dim)).astype(np.float32)
    # L2 normalize item factors so dot products are bounded / comparable
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    return F


def _sample_seq(rng: np.random.Generator, probs: np.ndarray, length: int) -> np.ndarray:
    return rng.choice(probs.shape[0], size=length, replace=True, p=probs)


def _generate_raw(cfg: ToyConfig, F: np.ndarray, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    n_items, k = F.shape

    hist = np.zeros((n_samples, cfg.hist_len), dtype=np.int64)
    anchor = np.zeros((n_samples, cfg.anchor_len), dtype=np.int64)
    cands = np.zeros((n_samples, cfg.n_candidates), dtype=np.int64)
    relevance = np.zeros((n_samples, cfg.n_candidates), dtype=np.float32)

    for n in range(n_samples):
        # latent user preference direction in factor space
        p = rng.normal(size=k).astype(np.float32)
        p /= (np.linalg.norm(p) + 1e-8)

        # history / anchor items reflect the user's preference (softmax sampling)
        aff = F @ p                                  # [n_items]
        logits = cfg.hist_temp * aff
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        hist[n] = _sample_seq(rng, probs, cfg.hist_len)
        anchor[n] = _sample_seq(rng, probs, cfg.anchor_len)

        # candidates: drawn uniformly (the items we must rank for this user)
        c = rng.choice(n_items, size=cfg.n_candidates, replace=False)
        cands[n] = c
        relevance[n] = (F[c] @ p)                    # ground-truth affinity

    return hist, anchor, cands, relevance


def _labels_from_relevance(relevance: np.ndarray, thr: Dict[str, float]) -> np.ndarray:
    """Build nested labels: order ⊂ cart ⊂ click by construction."""
    click = (relevance >= thr["click"]).astype(np.float32)
    cart = (relevance >= thr["cart"]).astype(np.float32) * click
    order = (relevance >= thr["order"]).astype(np.float32) * cart
    # stack -> [n, N, 3] in (click, cart, order) order
    return np.stack([click, cart, order], axis=-1).astype(np.float32)


def generate_dataset(cfg: ToyConfig, seed: int, n_train: int = 3000, n_test: int = 800) -> Dict:
    """Return train/test arrays sharing the same factor model and thresholds."""
    F = _factor_matrix(cfg)

    htr, atr, ctr, rtr = _generate_raw(cfg, F, n_train, seed)
    hte, ate, cte, rte = _generate_raw(cfg, F, n_test, seed + 777)

    # thresholds estimated on the train relevance distribution (then reused)
    flat = rtr.reshape(-1)
    thr = {
        "click": float(np.quantile(flat, 1.0 - cfg.click_q)),
        "cart": float(np.quantile(flat, 1.0 - cfg.cart_q)),
        "order": float(np.quantile(flat, 1.0 - cfg.order_q)),
    }

    train = {
        "hist": htr, "anchor": atr, "cands": ctr,
        "labels": _labels_from_relevance(rtr, thr),
    }
    test = {
        "hist": hte, "anchor": ate, "cands": cte,
        "labels": _labels_from_relevance(rte, thr),
    }
    return {"train": train, "test": test, "thresholds": thr}


def iterate_batches(split: Dict, batch_size: int, seed: int = 0, shuffle: bool = True):
    """Yield dicts of numpy arrays for one mini-batch."""
    n = split["hist"].shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        yield {
            "hist": split["hist"][b],
            "anchor": split["anchor"][b],
            "cands": split["cands"][b],
            "labels": split["labels"][b],
        }
