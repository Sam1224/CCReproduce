from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ClientData:
    client_id: int
    user_count: int
    interactions: List[Tuple[int, int]]  # (user_id, item_id)


@dataclass
class ToyFedDataset:
    vocab_size: int
    item_tokens: List[List[int]]
    clients: List[ClientData]
    test_next_item: Dict[Tuple[int, int], int]  # (client_id, user_id) -> item_id


def build_toy_federated_dataset(
    *,
    seed: int = 42,
    clients: int = 20,
    users_per_client: int = 50,
    items: int = 200,
    topics: int = 20,
    tokens_per_item: int = 12,
    interactions_per_user: int = 10,
    sparsity: float = 0.98,
) -> ToyFedDataset:
    """Create a toy federated recommender dataset.

    Design goal:
    - Each item has text tokens correlated with a latent topic.
    - Each user prefers a few topics.
    - Under high sparsity (few interactions), ID embeddings are noisy but text helps.

    `sparsity` controls the probability of dropping an interaction.
    """

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    vocab_size = topics * 50

    # Item text tokens: each item belongs to a topic, most tokens are topic-specific.
    item_topics = np_rng.integers(low=0, high=topics, size=(items,))
    item_tokens: List[List[int]] = []
    for item_id in range(items):
        topic = int(item_topics[item_id])
        topic_vocab = list(range(topic * 50, (topic + 1) * 50))
        tokens = [rng.choice(topic_vocab) for _ in range(tokens_per_item)]
        item_tokens.append(tokens)

    clients_data: List[ClientData] = []
    test_next_item: Dict[Tuple[int, int], int] = {}

    for client_id in range(clients):
        interactions: List[Tuple[int, int]] = []
        for user_id in range(users_per_client):
            # User prefers 3 topics.
            preferred = rng.sample(range(topics), k=3)

            # Generate a short sequence; last one reserved for test.
            seq: List[int] = []
            for _ in range(interactions_per_user + 1):
                t = rng.choice(preferred)
                # choose an item from that topic
                candidates = np.where(item_topics == t)[0]
                seq.append(int(rng.choice(list(candidates))))

            test_next_item[(client_id, user_id)] = seq[-1]

            for it in seq[:-1]:
                if rng.random() < sparsity:
                    continue
                interactions.append((user_id, it))

        clients_data.append(
            ClientData(client_id=client_id, user_count=users_per_client, interactions=interactions)
        )

    return ToyFedDataset(
        vocab_size=vocab_size,
        item_tokens=item_tokens,
        clients=clients_data,
        test_next_item=test_next_item,
    )
