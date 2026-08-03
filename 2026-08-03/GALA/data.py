from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import DataLoader, Dataset


CATEGORY_NAMES = [
    "snack",
    "drink",
    "meal",
    "dessert",
    "fruit",
    "grocery",
]


def _one_hot(index: int, size: int) -> torch.Tensor:
    vector = torch.zeros(size, dtype=torch.float32)
    vector[index] = 1.0
    return vector


@dataclass
class Catalog:
    text_features: torch.Tensor
    image_features: torch.Tensor
    id_features: torch.Tensor
    categories: torch.Tensor
    prices: torch.Tensor


class TripletDataset(Dataset):
    def __init__(self, queries: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):
        self.queries = queries
        self.positives = positives
        self.negatives = negatives

    def __len__(self) -> int:
        return self.queries.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "query": self.queries[index],
            "positive": self.positives[index],
            "negative": self.negatives[index],
        }


class RankingDataset(Dataset):
    def __init__(self, histories: torch.Tensor, queries: torch.Tensor, targets: torch.Tensor, rewards: torch.Tensor):
        self.histories = histories
        self.queries = queries
        self.targets = targets
        self.rewards = rewards

    def __len__(self) -> int:
        return self.queries.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "query": self.queries[index],
            "target": self.targets[index],
            "reward": self.rewards[index],
        }


def build_catalog(num_items: int = 48, feature_dim: int = 12, seed: int = 7) -> Catalog:
    generator = torch.Generator().manual_seed(seed)
    categories = torch.randint(0, len(CATEGORY_NAMES), (num_items,), generator=generator)
    prices = torch.rand(num_items, generator=generator) * 0.8 + 0.2

    text_features: List[torch.Tensor] = []
    image_features: List[torch.Tensor] = []
    id_features: List[torch.Tensor] = []
    for item_id in range(num_items):
        category = categories[item_id].item()
        category_basis = _one_hot(category, len(CATEGORY_NAMES))
        price_bucket = torch.tensor([
            prices[item_id],
            prices[item_id] ** 2,
            torch.sin(prices[item_id] * 3.14),
        ])
        noise_text = torch.randn(feature_dim - len(CATEGORY_NAMES) - 3, generator=generator) * 0.05
        noise_image = torch.randn(feature_dim - len(CATEGORY_NAMES) - 3, generator=generator) * 0.05
        text_features.append(torch.cat([category_basis, price_bucket, noise_text], dim=0))
        image_features.append(torch.cat([category_basis * 0.8, price_bucket.flip(0), noise_image], dim=0))
        id_features.append(torch.randn(feature_dim, generator=generator) * 0.2 + category_basis.repeat(2)[:feature_dim])

    return Catalog(
        text_features=torch.stack(text_features),
        image_features=torch.stack(image_features),
        id_features=torch.stack(id_features),
        categories=categories,
        prices=prices,
    )


def _compose_query(catalog: Catalog, target_item: int, generator: torch.Generator) -> torch.Tensor:
    category = catalog.categories[target_item].item()
    category_bias = _one_hot(category, len(CATEGORY_NAMES))
    price = catalog.prices[target_item]
    time_context = torch.tensor([
        1.0 if category in {0, 1, 3} else 0.2,
        1.0 if category in {2, 5} else 0.2,
        float(price > 0.55),
    ])
    noise = torch.randn(catalog.text_features.size(1) - len(CATEGORY_NAMES) - 3, generator=generator) * 0.04
    return torch.cat([category_bias, time_context, noise], dim=0)


def build_datasets(
    num_samples: int = 512,
    history_len: int = 5,
    seed: int = 11,
) -> Tuple[Catalog, TripletDataset, RankingDataset, RankingDataset]:
    catalog = build_catalog(seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    py_random = random.Random(seed + 1)

    queries: List[torch.Tensor] = []
    positives: List[torch.Tensor] = []
    negatives: List[torch.Tensor] = []
    histories: List[torch.Tensor] = []
    targets: List[int] = []
    rewards: List[float] = []

    num_items = catalog.text_features.size(0)
    for _ in range(num_samples):
        target_item = int(torch.randint(0, num_items, (1,), generator=generator).item())
        query = _compose_query(catalog, target_item, generator)
        negative_candidates = [
            item_id for item_id in range(num_items) if catalog.categories[item_id] != catalog.categories[target_item]
        ]
        negative_item = py_random.choice(negative_candidates)

        history_pool = [
            item_id for item_id in range(num_items) if catalog.categories[item_id] == catalog.categories[target_item]
        ]
        history = [py_random.choice(history_pool) for _ in range(history_len - 1)]
        history.append(target_item)

        positive_mm = 0.5 * (catalog.text_features[target_item] + catalog.image_features[target_item])
        negative_mm = 0.5 * (catalog.text_features[negative_item] + catalog.image_features[negative_item])
        price_match = 1.0 - torch.abs(catalog.prices[target_item] - 0.5).item()
        reward = 0.6 + 0.25 * price_match + 0.15 * float(catalog.categories[target_item].item() in {0, 2, 3})

        queries.append(query)
        positives.append(positive_mm)
        negatives.append(negative_mm)
        histories.append(torch.tensor(history, dtype=torch.long))
        targets.append(target_item)
        rewards.append(reward)

    split = int(num_samples * 0.8)
    triplet = TripletDataset(torch.stack(queries), torch.stack(positives), torch.stack(negatives))
    train_rank = RankingDataset(
        histories=torch.stack(histories[:split]),
        queries=torch.stack(queries[:split]),
        targets=torch.tensor(targets[:split], dtype=torch.long),
        rewards=torch.tensor(rewards[:split], dtype=torch.float32),
    )
    test_rank = RankingDataset(
        histories=torch.stack(histories[split:]),
        queries=torch.stack(queries[split:]),
        targets=torch.tensor(targets[split:], dtype=torch.long),
        rewards=torch.tensor(rewards[split:], dtype=torch.float32),
    )
    return catalog, triplet, train_rank, test_rank


def create_dataloaders(batch_size: int = 64, seed: int = 11) -> Tuple[Catalog, DataLoader, DataLoader, DataLoader]:
    catalog, triplet, train_rank, test_rank = build_datasets(seed=seed)
    return (
        catalog,
        DataLoader(triplet, batch_size=batch_size, shuffle=True),
        DataLoader(train_rank, batch_size=batch_size, shuffle=True),
        DataLoader(test_rank, batch_size=batch_size, shuffle=False),
    )
