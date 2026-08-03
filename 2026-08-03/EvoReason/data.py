from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import DataLoader, Dataset


PRIMITIVES = [
    "match_interest",
    "freshness_check",
    "quality_boost",
    "diversify_creator",
]


@dataclass
class Catalog:
    item_features: torch.Tensor
    creator_ids: torch.Tensor
    topic_ids: torch.Tensor
    quality_scores: torch.Tensor


class RecommendationDataset(Dataset):
    def __init__(self, histories: torch.Tensor, targets: torch.Tensor, primitive_labels: torch.Tensor):
        self.histories = histories
        self.targets = targets
        self.primitive_labels = primitive_labels

    def __len__(self) -> int:
        return self.histories.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "target": self.targets[index],
            "primitive_labels": self.primitive_labels[index],
        }


def build_catalog(num_items: int = 40, feature_dim: int = 10, seed: int = 23) -> Catalog:
    generator = torch.Generator().manual_seed(seed)
    topic_ids = torch.randint(0, 5, (num_items,), generator=generator)
    creator_ids = torch.randint(0, 8, (num_items,), generator=generator)
    quality_scores = torch.rand(num_items, generator=generator) * 0.7 + 0.3

    features: List[torch.Tensor] = []
    for item_id in range(num_items):
        topic = torch.nn.functional.one_hot(topic_ids[item_id], num_classes=5).float()
        creator = torch.nn.functional.one_hot(creator_ids[item_id], num_classes=8).float()[:3]
        stats = torch.tensor([
            quality_scores[item_id],
            float(topic_ids[item_id].item() in {1, 3}),
        ])
        noise = torch.randn(feature_dim - topic.numel() - creator.numel() - stats.numel(), generator=generator) * 0.06
        features.append(torch.cat([topic, creator, stats, noise], dim=0))

    return Catalog(
        item_features=torch.stack(features),
        creator_ids=creator_ids,
        topic_ids=topic_ids,
        quality_scores=quality_scores,
    )


def _primitive_sequence(catalog: Catalog, history: List[int], target: int) -> List[int]:
    history_topics = catalog.topic_ids[history]
    history_creators = catalog.creator_ids[history]
    top_topic = torch.mode(history_topics).values.item()
    primitives = [0 if catalog.topic_ids[target].item() == top_topic else 1]
    creator_repeat = int((history_creators == catalog.creator_ids[target]).any().item())
    primitives.append(3 if creator_repeat else 1)
    primitives.append(2 if catalog.quality_scores[target].item() > 0.7 else 0)
    return primitives


def build_datasets(num_samples: int = 640, history_len: int = 6, seed: int = 29) -> Tuple[Catalog, RecommendationDataset, RecommendationDataset]:
    catalog = build_catalog(seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    py_random = random.Random(seed + 1)

    histories: List[torch.Tensor] = []
    targets: List[int] = []
    primitive_labels: List[torch.Tensor] = []

    num_items = catalog.item_features.size(0)
    for _ in range(num_samples):
        dominant_topic = int(torch.randint(0, 5, (1,), generator=generator).item())
        history_pool = [item_id for item_id in range(num_items) if catalog.topic_ids[item_id].item() == dominant_topic]
        history = [py_random.choice(history_pool) for _ in range(history_len)]

        if py_random.random() < 0.7:
            candidate_pool = [item_id for item_id in range(num_items) if catalog.topic_ids[item_id].item() == dominant_topic]
        else:
            candidate_pool = list(range(num_items))
        target = py_random.choice(candidate_pool)
        histories.append(torch.tensor(history, dtype=torch.long))
        targets.append(target)
        primitive_labels.append(torch.tensor(_primitive_sequence(catalog, history, target), dtype=torch.long))

    split = int(num_samples * 0.8)
    train_dataset = RecommendationDataset(
        histories=torch.stack(histories[:split]),
        targets=torch.tensor(targets[:split], dtype=torch.long),
        primitive_labels=torch.stack(primitive_labels[:split]),
    )
    test_dataset = RecommendationDataset(
        histories=torch.stack(histories[split:]),
        targets=torch.tensor(targets[split:], dtype=torch.long),
        primitive_labels=torch.stack(primitive_labels[split:]),
    )
    return catalog, train_dataset, test_dataset


def create_dataloaders(batch_size: int = 64, seed: int = 29) -> Tuple[Catalog, DataLoader, DataLoader]:
    catalog, train_dataset, test_dataset = build_datasets(seed=seed)
    return (
        catalog,
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
