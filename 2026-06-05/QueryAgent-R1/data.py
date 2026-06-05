from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CatalogItem:
    item_id: int
    tokens: list[str]


@dataclass(frozen=True)
class UserExample:
    user_id: int
    history: list[int]
    target_query_tokens: list[str]
    purchase_set: set[int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_catalog(
    n_items: int,
    categories: list[str],
    colors: list[str],
    materials: list[str],
    brands: list[str],
) -> list[CatalogItem]:
    catalog: list[CatalogItem] = []
    for item_id in range(n_items):
        cat = random.choice(categories)
        color = random.choice(colors)
        material = random.choice(materials)
        brand = random.choice(brands)
        catalog.append(CatalogItem(item_id=item_id, tokens=[brand, cat, color, material]))
    return catalog


def _sample_pref(weights: np.ndarray) -> int:
    weights = weights / weights.sum()
    return int(np.random.choice(np.arange(len(weights)), p=weights))


def build_synthetic_users(
    *,
    n_users: int,
    catalog: list[CatalogItem],
    history_len: int,
    noise_click_rate: float,
) -> list[UserExample]:
    categories = sorted({it.tokens[1] for it in catalog})
    colors = sorted({it.tokens[2] for it in catalog})

    cat_to_items: dict[str, list[int]] = {c: [] for c in categories}
    color_to_items: dict[str, list[int]] = {c: [] for c in colors}
    for it in catalog:
        cat_to_items[it.tokens[1]].append(it.item_id)
        color_to_items[it.tokens[2]].append(it.item_id)

    users: list[UserExample] = []
    for user_id in range(n_users):
        cat_pref = np.random.dirichlet(np.ones(len(categories)))
        color_pref = np.random.dirichlet(np.ones(len(colors)))

        # Choose an intent (dominant category + dominant color)
        intent_cat = categories[int(cat_pref.argmax())]
        intent_color = colors[int(color_pref.argmax())]

        purchase_candidates = list(set(cat_to_items[intent_cat]) & set(color_to_items[intent_color]))
        if len(purchase_candidates) < 5:
            purchase_candidates = cat_to_items[intent_cat]
        purchase_set = set(random.sample(purchase_candidates, k=min(30, len(purchase_candidates))))

        history: list[int] = []
        for _ in range(history_len):
            if random.random() < noise_click_rate:
                history.append(random.randrange(len(catalog)))
                continue

            cat_idx = _sample_pref(cat_pref)
            color_idx = _sample_pref(color_pref)
            cat = categories[cat_idx]
            color = colors[color_idx]

            candidates = list(set(cat_to_items[cat]) & set(color_to_items[color]))
            if not candidates:
                candidates = cat_to_items[cat]
            history.append(random.choice(candidates))

        # Target query is simplified: [category, color]
        target_query_tokens = [intent_cat, intent_color]

        users.append(
            UserExample(
                user_id=user_id,
                history=history,
                target_query_tokens=target_query_tokens,
                purchase_set=purchase_set,
            )
        )

    return users


def train_val_test_split(examples: list[UserExample], seed: int = 0) -> tuple[list[UserExample], list[UserExample], list[UserExample]]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = [examples[i] for i in indices[:n_train]]
    val = [examples[i] for i in indices[n_train : n_train + n_val]]
    test = [examples[i] for i in indices[n_train + n_val :]]
    return train, val, test


def default_synthetic_setup(seed: int = 42) -> tuple[list[CatalogItem], list[UserExample], list[UserExample], list[UserExample]]:
    seed_everything(seed)

    categories = [
        "dress",
        "sneakers",
        "phone",
        "headphones",
        "laptop",
        "watch",
        "backpack",
        "chair",
        "lamp",
        "coffee",
    ]
    colors = ["black", "white", "red", "blue", "green", "pink", "gray", "brown"]
    materials = ["cotton", "leather", "plastic", "metal", "wood", "silk"]
    brands = ["nova", "zen", "orbit", "hype", "prime", "echo", "vela"]

    catalog = build_catalog(
        n_items=3000,
        categories=categories,
        colors=colors,
        materials=materials,
        brands=brands,
    )

    examples = build_synthetic_users(
        n_users=4000,
        catalog=catalog,
        history_len=20,
        noise_click_rate=0.2,
    )

    train, val, test = train_val_test_split(examples, seed=seed)
    return catalog, train, val, test
