from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

import torch
from torch.utils.data import Dataset


CATEGORIES = ["laptop", "tablet", "headphone"]
BRANDS = ["apple", "lenovo", "dell", "sony", "samsung", "anker"]
VOCAB = {
    token: index
    for index, token in enumerate(
        [
            "budget",
            "cheap",
            "premium",
            "laptop",
            "tablet",
            "headphone",
            "gaming",
            "office",
            "travel",
            "exclude_apple",
            "exclude_sony",
            "exclude_samsung",
            "need_ram16",
            "need_storage512",
            "relax_budget",
            "reverse_exclusion",
        ]
    )
}


@dataclass
class CatalogItem:
    item_id: str
    category: str
    brand: str
    price: float
    ram_gb: int
    storage_gb: int
    quality: float


@dataclass
class TurnSpec:
    query_tokens: list[str]
    category: str
    budget_max: float
    excluded_brand: str | None
    required_ram: int
    required_storage: int
    relax_budget: bool = False
    reverse_exclusion: bool = False


class SessionMemory:
    def __init__(self) -> None:
        self.category: str | None = None
        self.budget_max: float = 0.0
        self.excluded_brands: set[str] = set()
        self.required_ram: int = 0
        self.required_storage: int = 0

    def update(self, turn: TurnSpec) -> None:
        if turn.category:
            self.category = turn.category
        if turn.budget_max:
            if self.budget_max:
                self.budget_max = min(self.budget_max, turn.budget_max)
            else:
                self.budget_max = turn.budget_max
        if turn.excluded_brand:
            if turn.reverse_exclusion and turn.excluded_brand in self.excluded_brands:
                self.excluded_brands.remove(turn.excluded_brand)
            else:
                self.excluded_brands.add(turn.excluded_brand)
        self.required_ram = max(self.required_ram, turn.required_ram)
        self.required_storage = max(self.required_storage, turn.required_storage)
        if turn.relax_budget:
            self.budget_max += 150.0

    def as_vector(self) -> torch.Tensor:
        category_vec = [1.0 if self.category == category else 0.0 for category in CATEGORIES]
        brand_mask = [1.0 if brand in self.excluded_brands else 0.0 for brand in BRANDS]
        numeric = [
            self.budget_max / 2000.0,
            self.required_ram / 32.0,
            self.required_storage / 1024.0,
        ]
        return torch.tensor(category_vec + brand_mask + numeric, dtype=torch.float32)


def build_catalog(seed: int = 7) -> list[CatalogItem]:
    rng = Random(seed)
    catalog: list[CatalogItem] = []
    for category in CATEGORIES:
        for brand in BRANDS:
            for variant in range(8):
                if category == "laptop":
                    ram = rng.choice([8, 16, 32])
                    storage = rng.choice([256, 512, 1024])
                    base = 650
                elif category == "tablet":
                    ram = rng.choice([4, 8, 12])
                    storage = rng.choice([128, 256, 512])
                    base = 380
                else:
                    ram = rng.choice([0, 0, 0])
                    storage = rng.choice([0, 0, 0])
                    base = 90
                quality = round(rng.uniform(0.55, 0.98), 4)
                price = base + variant * rng.randint(20, 90) + quality * 180 + ram * 6 + storage * 0.18
                catalog.append(
                    CatalogItem(
                        item_id=f"{category[:2]}-{brand[:2]}-{variant}",
                        category=category,
                        brand=brand,
                        price=round(price, 2),
                        ram_gb=ram,
                        storage_gb=storage,
                        quality=quality,
                    )
                )
    return catalog


def turn_to_bow(turn: TurnSpec) -> torch.Tensor:
    vector = torch.zeros(len(VOCAB), dtype=torch.float32)
    for token in turn.query_tokens:
        if token in VOCAB:
            vector[VOCAB[token]] += 1.0
    return vector


def candidate_to_vector(item: CatalogItem) -> torch.Tensor:
    category_vec = [1.0 if item.category == category else 0.0 for category in CATEGORIES]
    brand_vec = [1.0 if item.brand == brand else 0.0 for brand in BRANDS]
    numeric = [
        item.price / 2000.0,
        item.ram_gb / 32.0,
        item.storage_gb / 1024.0,
        item.quality,
    ]
    return torch.tensor(category_vec + brand_vec + numeric, dtype=torch.float32)


def retrieve_candidates(catalog: Iterable[CatalogItem], memory: SessionMemory, top_k: int = 8) -> list[CatalogItem]:
    grounded = [item for item in catalog if memory.category in (None, item.category)]
    grounded = [item for item in grounded if item.brand not in memory.excluded_brands]
    if memory.budget_max:
        grounded = [item for item in grounded if item.price <= memory.budget_max]
    if memory.required_ram:
        grounded = [item for item in grounded if item.ram_gb >= memory.required_ram]
    if memory.required_storage:
        grounded = [item for item in grounded if item.storage_gb >= memory.required_storage]
    return sorted(grounded, key=lambda item: (item.quality, -item.price), reverse=True)[:top_k]


def oracle_target(candidates: list[CatalogItem], memory: SessionMemory) -> str:
    def utility(item: CatalogItem) -> float:
        price_gap = 0.0
        if memory.budget_max:
            price_gap = abs(memory.budget_max - item.price) / 2000.0
        spec_bonus = 0.2 * (item.ram_gb >= memory.required_ram) + 0.2 * (item.storage_gb >= memory.required_storage)
        return item.quality + spec_bonus - price_gap

    return max(candidates, key=utility).item_id


def build_turn_specs(seed: int = 13, num_sessions: int = 180) -> list[list[TurnSpec]]:
    rng = Random(seed)
    sessions: list[list[TurnSpec]] = []
    for _ in range(num_sessions):
        category = rng.choice(CATEGORIES)
        if category == "laptop":
            budget = rng.choice([900, 1100, 1400])
            required_ram = rng.choice([8, 16])
            required_storage = rng.choice([256, 512])
        elif category == "tablet":
            budget = rng.choice([450, 650, 850])
            required_ram = rng.choice([4, 8])
            required_storage = rng.choice([128, 256])
        else:
            budget = rng.choice([120, 180, 260])
            required_ram = 0
            required_storage = 0
        brand = rng.choice(BRANDS)
        tokens = ["budget", category]
        if required_ram >= 16:
            tokens.append("need_ram16")
        if required_storage >= 512:
            tokens.append("need_storage512")
        session = [
            TurnSpec(tokens, category, budget, brand, required_ram, required_storage),
            TurnSpec(["office" if category != "headphone" else "travel"], category, 0.0, None, 0, 0),
        ]
        if rng.random() < 0.45:
            session.append(TurnSpec(["reverse_exclusion"], category, 0.0, brand, 0, 0, reverse_exclusion=True))
        if rng.random() < 0.35:
            session.append(TurnSpec(["relax_budget"], category, 0.0, None, 0, 0, relax_budget=True))
        sessions.append(session)
    return sessions


class RankingRowDataset(Dataset):
    def __init__(self, catalog: list[CatalogItem], sessions: list[list[TurnSpec]]) -> None:
        self.rows: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for turns in sessions:
            memory = SessionMemory()
            for turn in turns:
                memory.update(turn)
                candidates = retrieve_candidates(catalog, memory)
                if not candidates:
                    continue
                target_item_id = oracle_target(candidates, memory)
                turn_vec = turn_to_bow(turn)
                memory_vec = memory.as_vector()
                for candidate in candidates:
                    candidate_vec = candidate_to_vector(candidate)
                    features = torch.cat([turn_vec, memory_vec])
                    label = 1.0 if candidate.item_id == target_item_id else 0.0
                    self.rows.append((features, candidate_vec, label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate_rows(batch):
    query_feats = torch.stack([row[0] for row in batch])
    candidate_feats = torch.stack([row[1] for row in batch])
    labels = torch.tensor([row[2] for row in batch], dtype=torch.float32)
    return query_feats, candidate_feats, labels


def benchmark_sessions() -> list[list[TurnSpec]]:
    return [
        [
            TurnSpec(["budget", "laptop", "need_ram16"], "laptop", 1100, "apple", 16, 512),
            TurnSpec(["office"], "laptop", 0.0, None, 0, 0),
        ],
        [
            TurnSpec(["budget", "tablet"], "tablet", 650, "samsung", 8, 256),
            TurnSpec(["reverse_exclusion"], "tablet", 0.0, "samsung", 0, 0, reverse_exclusion=True),
        ],
        [
            TurnSpec(["budget", "headphone"], "headphone", 140, "sony", 0, 0),
            TurnSpec(["relax_budget"], "headphone", 0.0, None, 0, 0, relax_budget=True),
        ],
    ]
