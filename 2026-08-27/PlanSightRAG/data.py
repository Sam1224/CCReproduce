from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PlanPage:
    page_id: int
    patch_features: torch.Tensor
    rule_value: float
    label: int


class PlanSightDataset(Dataset):
    def __init__(self, pages: List[PlanPage], queries: List[Tuple[torch.Tensor, int, float]]) -> None:
        self.pages = pages
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_vec, gold_page, proposed_value = self.queries[idx]
        return {
            "query_vec": query_vec.float(),
            "gold_page": torch.tensor(gold_page, dtype=torch.long),
            "proposed_value": torch.tensor(proposed_value, dtype=torch.float32),
        }


def build_corpus(seed: int = 19, num_pages: int = 96, patches: int = 12, dim: int = 32, rules: int = 8) -> List[PlanPage]:
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    rule_proto = torch.randn(rules, dim, generator=gen)
    pages: List[PlanPage] = []
    for page_id in range(num_pages):
        label = page_id % rules
        patch = rule_proto[label].unsqueeze(0) + 0.35 * torch.randn(patches, dim, generator=gen)
        pages.append(PlanPage(page_id=page_id, patch_features=patch, rule_value=1.0 + 0.2 * label + rng.random() * 0.2, label=label))
    return pages


def build_queries(pages: List[PlanPage], seed: int, n: int, dim: int = 32) -> List[Tuple[torch.Tensor, int, float]]:
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    queries = []
    for _ in range(n):
        page = rng.choice(pages)
        query_vec = page.patch_features.mean(dim=0) + 0.25 * torch.randn(dim, generator=gen)
        proposed = page.rule_value + rng.choice([-0.25, -0.05, 0.05, 0.25])
        queries.append((query_vec, page.page_id, proposed))
    return queries


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_dataloaders(batch_size: int = 64):
    pages = build_corpus()
    train = PlanSightDataset(pages, build_queries(pages, 20, 1200))
    val = PlanSightDataset(pages, build_queries(pages, 21, 250))
    test = PlanSightDataset(pages, build_queries(pages, 22, 250))
    kwargs = {"batch_size": batch_size, "collate_fn": collate_fn}
    return pages, DataLoader(train, shuffle=True, **kwargs), DataLoader(val, **kwargs), DataLoader(test, **kwargs)
