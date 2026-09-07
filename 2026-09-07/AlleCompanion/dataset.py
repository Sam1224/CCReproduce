from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class ComCatMap:
    """Maintain complementary category priors from multiple weak sources."""

    def __init__(self, num_categories: int = 16, seed: int = 23) -> None:
        generator = torch.Generator().manual_seed(seed)
        expert = torch.eye(num_categories).roll(shifts=1, dims=1) * 0.6
        llm = torch.rand(num_categories, num_categories, generator=generator) * 0.3
        stats = torch.rand(num_categories, num_categories, generator=generator) * 0.3
        self.matrix = (expert + llm + stats).clamp(max=1.0)
        self.matrix.fill_diagonal_(0.05)

    def score(self, source_category: torch.Tensor, target_category: torch.Tensor) -> torch.Tensor:
        return self.matrix[source_category, target_category]

    def valid_targets(self, source_category: int, threshold: float = 0.45) -> torch.Tensor:
        return torch.where(self.matrix[source_category] >= threshold)[0]


class ToyComplementDataset(Dataset):
    def __init__(self, num_items: int = 512, num_pairs: int = 4096, num_categories: int = 16, feature_dim: int = 64, seed: int = 29) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.num_items = num_items
        self.num_categories = num_categories
        self.feature_dim = feature_dim
        self.item_features = torch.randn(num_items, feature_dim, generator=generator)
        self.item_categories = torch.randint(0, num_categories, (num_items,), generator=generator)
        self.comcat = ComCatMap(num_categories, seed=seed)
        queries = torch.randint(0, num_items, (num_pairs,), generator=generator)
        candidates = torch.randint(0, num_items, (num_pairs,), generator=generator)
        category_score = self.comcat.score(self.item_categories[queries], self.item_categories[candidates])
        semantic_score = torch.cosine_similarity(self.item_features[queries], self.item_features[candidates], dim=-1).clamp_min(0)
        probability = (0.65 * category_score + 0.35 * semantic_score).clamp(0, 1)
        labels = torch.bernoulli(probability, generator=generator)
        self.queries = queries
        self.candidates = candidates
        self.labels = labels.float()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q = self.queries[idx]
        c = self.candidates[idx]
        return {
            "query_feature": self.item_features[q],
            "candidate_feature": self.item_features[c],
            "query_category": self.item_categories[q],
            "candidate_category": self.item_categories[c],
            "label": self.labels[idx],
        }
