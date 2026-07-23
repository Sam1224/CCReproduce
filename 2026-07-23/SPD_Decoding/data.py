from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Catalog:
    item_features: torch.Tensor
    item_categories: torch.Tensor
    creator_groups: torch.Tensor
    business_scores: torch.Tensor


class SyntheticSlateDataset(Dataset):
    def __init__(
        self,
        user_states: torch.Tensor,
        targets: torch.Tensor,
        target_constraints: torch.Tensor,
    ) -> None:
        self.user_states = user_states.float()
        self.targets = targets.long()
        self.target_constraints = target_constraints.float()

    def __len__(self) -> int:
        return self.user_states.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "user_state": self.user_states[index],
            "targets": self.targets[index],
            "constraints": self.target_constraints[index],
        }


def _oracle_slate(
    user_state: torch.Tensor,
    catalog: Catalog,
    slate_size: int,
    min_tail_ratio: float,
    min_category_count: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    item_features = catalog.item_features
    relevance = item_features @ user_state
    chosen: List[int] = []
    category_hist: Dict[int, int] = {}
    tail_count = 0
    lambdas = torch.tensor([1.1, 0.9], dtype=torch.float32)

    for step in range(slate_size):
        logits = relevance.clone()
        for selected in chosen:
            logits[selected] = -1e9

        tail_signal = (catalog.creator_groups == 1).float()
        diversity_signal = torch.ones_like(tail_signal)
        if category_hist:
            seen_categories = torch.tensor(list(category_hist.keys()), dtype=torch.long)
            diversity_signal = (~torch.isin(catalog.item_categories, seen_categories)).float()

        adjusted = (
            logits
            + lambdas[0] * (tail_signal - min_tail_ratio)
            + lambdas[1] * (diversity_signal - float(min_category_count > len(category_hist)))
            + 0.2 * catalog.business_scores
        )
        probs = torch.softmax(adjusted / temperature, dim=0)
        next_item = torch.multinomial(probs, num_samples=1).item()
        chosen.append(next_item)

        category = int(catalog.item_categories[next_item].item())
        category_hist[category] = category_hist.get(category, 0) + 1
        tail_count += int(catalog.creator_groups[next_item].item())

        current_tail_ratio = tail_count / len(chosen)
        category_coverage = len(category_hist)
        violations = torch.tensor(
            [max(0.0, min_tail_ratio - current_tail_ratio), max(0.0, min_category_count - category_coverage)],
            dtype=torch.float32,
        )
        lambdas = torch.relu(lambdas + 0.7 * violations)

    targets = torch.tensor(chosen, dtype=torch.long)
    constraints = torch.tensor([min_tail_ratio, float(min_category_count)], dtype=torch.float32)
    return targets, constraints


def build_synthetic_catalog(
    num_items: int = 90,
    feature_dim: int = 16,
    num_categories: int = 6,
    seed: int = 7,
) -> Catalog:
    generator = torch.Generator().manual_seed(seed)
    item_features = torch.randn(num_items, feature_dim, generator=generator)
    item_categories = torch.randint(0, num_categories, (num_items,), generator=generator)
    creator_groups = (torch.rand(num_items, generator=generator) > 0.72).long()
    business_scores = torch.rand(num_items, generator=generator) * 2 - 1
    return Catalog(
        item_features=item_features,
        item_categories=item_categories,
        creator_groups=creator_groups,
        business_scores=business_scores,
    )


def build_dataset(
    num_users: int,
    catalog: Catalog,
    slate_size: int,
    seed: int,
) -> SyntheticSlateDataset:
    generator = torch.Generator().manual_seed(seed)
    feature_dim = catalog.item_features.size(1)
    user_states = torch.randn(num_users, feature_dim, generator=generator)

    targets = []
    constraints = []
    for index in range(num_users):
        min_tail_ratio = 0.2 + 0.2 * (index % 3)
        min_category_count = 2 + (index % 2)
        slate, slate_constraints = _oracle_slate(
            user_states[index],
            catalog,
            slate_size=slate_size,
            min_tail_ratio=min_tail_ratio,
            min_category_count=min_category_count,
            temperature=0.9,
        )
        targets.append(slate)
        constraints.append(slate_constraints)

    return SyntheticSlateDataset(
        user_states=user_states,
        targets=torch.stack(targets),
        target_constraints=torch.stack(constraints),
    )


def create_dataloaders(
    batch_size: int = 32,
    slate_size: int = 5,
    seed: int = 7,
) -> Tuple[Catalog, DataLoader, DataLoader]:
    catalog = build_synthetic_catalog(seed=seed)
    train_dataset = build_dataset(640, catalog, slate_size, seed=seed + 1)
    test_dataset = build_dataset(160, catalog, slate_size, seed=seed + 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return catalog, train_loader, test_loader


def slate_metrics(
    slates: torch.Tensor,
    user_states: torch.Tensor,
    constraints: torch.Tensor,
    catalog: Catalog,
) -> Dict[str, float]:
    batch_size, slate_size = slates.shape
    gathered_features = catalog.item_features[slates]
    relevance = (gathered_features * user_states.unsqueeze(1)).sum(dim=-1).mean().item()
    business = catalog.business_scores[slates].mean().item()
    tail_ratio = catalog.creator_groups[slates].float().mean().item()

    categories = catalog.item_categories[slates]
    unique_counts = []
    tail_violations = []
    category_violations = []
    for index in range(batch_size):
        unique_count = torch.unique(categories[index]).numel()
        unique_counts.append(float(unique_count))
        tail_violations.append(max(0.0, constraints[index, 0].item() - catalog.creator_groups[slates[index]].float().mean().item()))
        category_violations.append(max(0.0, constraints[index, 1].item() - unique_count))

    return {
        "relevance": relevance,
        "business": business,
        "tail_ratio": tail_ratio,
        "category_coverage": sum(unique_counts) / len(unique_counts),
        "tail_violation": sum(tail_violations) / len(tail_violations),
        "category_violation": sum(category_violations) / len(category_violations),
    }
