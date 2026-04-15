from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class QueryExample:
    query_tokens: List[int]
    candidate_item_ids: List[int]
    graded_relevance: List[int]  # 0..4, aligned with candidate_item_ids
    prior_score: List[float]  # 0..1
    engagement_score: List[float]  # 0..1 (sparse)


@dataclass
class ToySponsoredDataset:
    vocab_size: int
    item_tokens: List[List[int]]
    train: List[QueryExample]
    test: List[QueryExample]


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def build_toy_dataset(
    *,
    seed: int = 42,
    items: int = 400,
    queries: int = 2000,
    candidates: int = 80,
    topics: int = 30,
    tokens_per_item: int = 10,
    tokens_per_query: int = 6,
    channels: int = 3,
    train_ratio: float = 0.8,
) -> ToySponsoredDataset:
    """Synthetic sponsored-search retrieval dataset.

    Construct signals analogous to the paper:
    - graded semantic relevance r(q,i) in {0..4}
    - multi-channel retrieval prior p(q,i)
    - engagement e(q,i) that is noisy & structurally sparse

    We intentionally make engagement biased by popularity/exposure.
    """

    rng = np.random.default_rng(seed)

    vocab_size = topics * 50

    item_topics = rng.integers(0, topics, size=(items,))
    item_popularity = rng.uniform(0.0, 1.0, size=(items,))

    item_tokens: List[List[int]] = []
    for item_id in range(items):
        t = int(item_topics[item_id])
        topic_vocab = np.arange(t * 50, (t + 1) * 50)
        tokens = rng.choice(topic_vocab, size=(tokens_per_item,), replace=True).tolist()
        item_tokens.append(tokens)

    examples: List[QueryExample] = []
    for _ in range(queries):
        q_topic = int(rng.integers(0, topics))
        q_vocab = np.arange(q_topic * 50, (q_topic + 1) * 50)
        q_tokens = rng.choice(q_vocab, size=(tokens_per_query,), replace=True).tolist()

        # Candidate pool mixes relevant & irrelevant.
        # We oversample high-pop items to mimic exposure bias.
        cand_ids = rng.choice(
            np.arange(items),
            size=(candidates,),
            replace=False,
            p=item_popularity / item_popularity.sum(),
        ).tolist()

        rels: List[int] = []
        priors: List[float] = []
        engages: List[float] = []

        for iid in cand_ids:
            topic_match = 1.0 if item_topics[iid] == q_topic else 0.0
            base = 0.6 * topic_match + 0.4 * (1 - abs(item_popularity[iid] - 0.6))
            base = float(np.clip(base, 0.0, 1.0))

            # Graded relevance label (teacher output).
            r = int(np.digitize(base, bins=[0.2, 0.4, 0.6, 0.8]))  # 0..4
            rels.append(r)

            # Multi-channel retrieval prior: high if multiple channels rank it high.
            channel_scores = []
            for _c in range(channels):
                # Each channel is correlated with relevance but noisy.
                channel_scores.append(base + rng.normal(0, 0.12))
            prior = float(np.mean(channel_scores))
            prior = float(np.clip(prior, 0.0, 1.0))
            priors.append(prior)

            # Engagement: sparse and biased.
            # Impression probability depends on auction/budget-like randomness and prior.
            impression = rng.random() < (0.15 + 0.35 * prior)
            if not impression:
                engages.append(0.0)
            else:
                # Engagement is influenced by relevance + popularity + noise.
                e = 0.45 * base + 0.45 * float(item_popularity[iid]) + 0.10 * rng.random()
                engages.append(float(np.clip(e, 0.0, 1.0)))

        # Normalize engagement within-query (paper uses query-wise max + sigmoid smoothing).
        max_e = max(engages) if max(engages) > 0 else 1.0
        engages = [float(1 / (1 + math.exp(-8 * ((e / max_e) - 0.5)))) for e in engages]

        examples.append(
            QueryExample(
                query_tokens=q_tokens,
                candidate_item_ids=cand_ids,
                graded_relevance=rels,
                prior_score=priors,
                engagement_score=engages,
            )
        )

    rng.shuffle(examples)
    split = int(len(examples) * train_ratio)
    return ToySponsoredDataset(
        vocab_size=vocab_size,
        item_tokens=item_tokens,
        train=examples[:split],
        test=examples[split:],
    )


def build_target_distribution(
    ex: QueryExample,
    *,
    mode: str,
    w_prior: float = 0.5,
    w_eng: float = 0.8,
    hard_neg: float = 0.6,
    temperature: float = 1.0,
) -> np.ndarray:
    """Create a listwise target distribution over candidates.

    Modes:
    - eng_only: only engagement score
    - rel_only: relevance (primary) + prior
    - rel_eng: relevance (primary) + prior + engagement among relevant items

    For irrelevant items, penalize high-prior items as hard negatives.
    """

    rel = np.asarray(ex.graded_relevance, dtype=np.float32) / 4.0
    prior = np.asarray(ex.prior_score, dtype=np.float32)
    eng = np.asarray(ex.engagement_score, dtype=np.float32)

    relevant = (rel >= 0.5).astype(np.float32)

    if mode == "eng_only":
        score = eng
    elif mode == "rel_only":
        score = rel + w_prior * prior - hard_neg * (1 - relevant) * prior
    elif mode == "rel_eng":
        score = rel + w_prior * prior + w_eng * relevant * eng - hard_neg * (1 - relevant) * prior
    else:
        raise ValueError(f"unknown mode: {mode}")

    return _softmax(score, temperature=temperature)
