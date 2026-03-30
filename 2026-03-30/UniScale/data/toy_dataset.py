from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Sample:
    user_id: int
    query_id: int
    item_id: int
    domain_id: int
    user_tokens: np.ndarray  # (U,)
    query_tokens: np.ndarray  # (Q,)
    item_tokens: np.ndarray  # (I,)
    click: int
    purchase: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_toy_logs(
    *,
    seed: int = 7,
    num_users: int = 200,
    num_items: int = 2000,
    num_queries: int = 1000,
    domains: int = 3,
    token_vocab: int = 4096,
    user_token_len: int = 6,
    query_token_len: int = 8,
    item_token_len: int = 10,
    interactions: int = 50000,
) -> Tuple[List[Sample], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Generate a toy e-commerce search log.

    Returns:
      - samples: list of interaction samples (positive + negative candidates not included)
      - item_vectors: latent item embedding for sampling / scoring
      - query_vectors: latent query embedding for sampling / scoring

    Label generation:
      - click probability = sigmoid(q·i + u·i + noise)
      - purchase probability = click * sigmoid(q·i + u·i - 0.5 + noise)

    purchase => click is enforced.
    """

    rng = np.random.default_rng(seed)

    latent_dim = 32
    user_latent = rng.standard_normal((num_users, latent_dim)).astype(np.float32)
    item_latent = rng.standard_normal((num_items, latent_dim)).astype(np.float32)
    query_latent = rng.standard_normal((num_queries, latent_dim)).astype(np.float32)

    samples: List[Sample] = []

    # Pre-generate tokens
    user_tokens = rng.integers(0, token_vocab, size=(num_users, user_token_len), dtype=np.int64)
    item_tokens = rng.integers(0, token_vocab, size=(num_items, item_token_len), dtype=np.int64)
    query_tokens = rng.integers(0, token_vocab, size=(num_queries, query_token_len), dtype=np.int64)

    for _ in range(interactions):
        user_id = int(rng.integers(0, num_users))
        query_id = int(rng.integers(0, num_queries))
        domain_id = int(rng.integers(0, domains))

        # Choose an exposed item with preference bias
        scores = query_latent[query_id] @ item_latent.T + user_latent[user_id] @ item_latent.T
        # Softmax sampling for exposure bias
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        item_id = int(rng.choice(np.arange(num_items), p=probs))

        base = float(query_latent[query_id] @ item_latent[item_id] + user_latent[user_id] @ item_latent[item_id])
        click_p = float(_sigmoid(np.array(base + rng.normal(scale=0.8))))
        click = int(rng.random() < click_p)

        purchase = 0
        if click:
            purchase_p = float(_sigmoid(np.array(base - 0.5 + rng.normal(scale=0.8))))
            purchase = int(rng.random() < purchase_p)

        samples.append(
            Sample(
                user_id=user_id,
                query_id=query_id,
                item_id=item_id,
                domain_id=domain_id,
                user_tokens=user_tokens[user_id].copy(),
                query_tokens=query_tokens[query_id].copy(),
                item_tokens=item_tokens[item_id].copy(),
                click=click,
                purchase=purchase,
            )
        )

    item_vectors = {i: item_latent[i].copy() for i in range(num_items)}
    query_vectors = {q: query_latent[q].copy() for q in range(num_queries)}
    return samples, item_vectors, query_vectors
