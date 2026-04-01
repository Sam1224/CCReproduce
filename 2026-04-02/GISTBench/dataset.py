from __future__ import annotations

import random
from dataclasses import dataclass


TAXONOMY = {
    "root": {
        "tech": ["ai", "programming", "gadgets"],
        "fashion": ["shoes", "streetwear", "cosmetics"],
        "food": ["coffee", "dessert", "cooking"],
        "sports": ["football", "running", "basketball"],
        "travel": ["hotels", "flights", "city-guides"],
    }
}


@dataclass(frozen=True)
class Interaction:
    item_text: str
    category: str
    signal: str
    weight: float


@dataclass(frozen=True)
class SyntheticUser:
    user_id: str
    history: list[Interaction]
    gt_interests: list[str]
    survey_score: float


SIGNALS = [
    ("watch", 1.0),
    ("like", 2.0),
    ("dislike", -2.0),
    ("skip", -0.5),
]


def _sample_leaf_category(rng: random.Random) -> str:
    parent = rng.choice(list(TAXONOMY["root"].keys()))
    leaf = rng.choice(TAXONOMY["root"][parent])
    return f"{parent}/{leaf}"


def _interest_from_category(category: str) -> str:
    return category.split("/")[0]


def build_synthetic_users(*, num_users: int = 80, seed: int = 0) -> list[SyntheticUser]:
    rng = random.Random(seed)
    users: list[SyntheticUser] = []

    for user_idx in range(num_users):
        base_interest = rng.choice(list(TAXONOMY["root"].keys()))
        history: list[Interaction] = []

        for _ in range(rng.randint(40, 80)):
            if rng.random() < 0.65:
                parent = base_interest
                leaf = rng.choice(TAXONOMY["root"][parent])
                category = f"{parent}/{leaf}"
            else:
                category = _sample_leaf_category(rng)

            signal, w = rng.choice(SIGNALS)
            item_text = f"video about {category}"
            history.append(Interaction(item_text=item_text, category=category, signal=signal, weight=w))

        scores: dict[str, float] = {}
        for interaction in history:
            parent = _interest_from_category(interaction.category)
            scores[parent] = scores.get(parent, 0.0) + interaction.weight

        gt_sorted = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        gt_interests = [k for k, _ in gt_sorted[:3]]

        survey_score = max(0.0, min(1.0, 0.2 + 0.15 * scores.get(base_interest, 0.0) / 20.0 + rng.random() * 0.1))

        users.append(
            SyntheticUser(
                user_id=f"u{user_idx:04d}",
                history=history,
                gt_interests=gt_interests,
                survey_score=survey_score,
            )
        )

    return users
