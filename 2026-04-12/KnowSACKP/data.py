from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class UserCase:
    history: List[int]
    candidates: List[int]
    ground_truth: int


@dataclass
class ToyRecDataset:
    vocab_size: int
    item_tokens: List[List[int]]
    item_popularity: np.ndarray
    item_latent: np.ndarray
    user_cases: List[UserCase]


def build_toy_rec_dataset(
    *,
    seed: int = 42,
    items: int = 500,
    users: int = 2000,
    topics: int = 40,
    tokens_per_item: int = 12,
    history_len: int = 10,
    candidates: int = 20,
) -> ToyRecDataset:
    rng = np.random.default_rng(seed)

    vocab_size = topics * 50

    # Ensure each topic has at least one item (important for small toy sizes).
    repeats = int(np.ceil(items / topics))
    item_topics = np.repeat(np.arange(topics), repeats)[:items]
    rng.shuffle(item_topics)
    item_popularity = rng.uniform(0.0, 1.0, size=(items,))

    # Latent factors per topic.
    topic_latent = rng.normal(0, 1, size=(topics, 32))
    item_latent = topic_latent[item_topics] + 0.1 * rng.normal(0, 1, size=(items, 32))

    item_tokens: List[List[int]] = []
    for iid in range(items):
        t = int(item_topics[iid])
        vocab = np.arange(t * 50, (t + 1) * 50)
        tokens = rng.choice(vocab, size=(tokens_per_item,), replace=True).tolist()
        item_tokens.append(tokens)

    user_cases: List[UserCase] = []
    for _ in range(users):
        pref = rng.choice(np.arange(topics), size=(3,), replace=False)
        # Build history.
        history_items: List[int] = []
        for _h in range(history_len):
            t = int(rng.choice(pref))
            candidates_t = np.where(item_topics == t)[0]
            history_items.append(int(rng.choice(candidates_t)))

        # Ground truth: from preferred topics but skew to tail items.
        t = int(rng.choice(pref))
        cand_t = np.where(item_topics == t)[0]
        # prefer tail
        tail_probs = 1 - item_popularity[cand_t]
        tail_probs = tail_probs / tail_probs.sum()
        gt = int(rng.choice(cand_t, p=tail_probs))

        # Candidates: include gt + other distractors (some popular, some same-topic).
        cand_ids = set([gt])
        while len(cand_ids) < candidates:
            if rng.random() < 0.5:
                # popular distractor
                cand_ids.add(int(rng.choice(np.argsort(-item_popularity)[: items // 5])))
            else:
                cand_ids.add(int(rng.integers(0, items)))

        user_cases.append(UserCase(history=history_items, candidates=list(cand_ids), ground_truth=gt))

    return ToyRecDataset(
        vocab_size=vocab_size,
        item_tokens=item_tokens,
        item_popularity=item_popularity,
        item_latent=item_latent,
        user_cases=user_cases,
    )
