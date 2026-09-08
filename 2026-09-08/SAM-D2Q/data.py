from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

CATEGORIES = ["dress", "sneaker", "backpack", "headphone", "lamp", "chair"]
VISIBLE_STYLES = ["casual", "formal", "sport", "minimal", "retro", "travel"]
VISUAL_ATTRIBUTES = [
    "red",
    "blue",
    "black",
    "white",
    "leather",
    "canvas",
    "silk",
    "wooden",
    "striped",
    "oversized",
    "wireless",
    "foldable",
]
DEMAND_TOKENS = [
    "gift",
    "commute",
    "wedding",
    "gaming",
    "outdoor",
    "office",
    "travel",
    "lightweight",
    "premium",
    "space-saving",
    "creator",
    "streaming",
]

TITLE_VOCAB = CATEGORIES + VISIBLE_STYLES + VISUAL_ATTRIBUTES + DEMAND_TOKENS
EXPANSION_VOCAB = VISUAL_ATTRIBUTES + DEMAND_TOKENS
PAD_ID = 0
TOKEN_TO_ID = {token: index + 1 for index, token in enumerate(TITLE_VOCAB)}
EXPANSION_TO_ID = {token: index for index, token in enumerate(EXPANSION_VOCAB)}
VOCAB_SIZE = len(TITLE_VOCAB) + 1
IMAGE_DIM = len(VISUAL_ATTRIBUTES)
MAX_TITLE_LEN = 6

COMMERCIAL_WEIGHTS = {
    "gift": 1.2,
    "premium": 1.15,
    "travel": 1.1,
    "gaming": 1.05,
    "streaming": 1.05,
    "creator": 1.05,
    "office": 1.0,
    "commute": 0.95,
    "outdoor": 0.95,
    "lightweight": 0.9,
    "space-saving": 0.9,
    "wedding": 0.9,
}


@dataclass(frozen=True)
class Product:
    category: str
    title_tokens: List[str]
    visual_attributes: List[str]
    demand_tokens: List[str]
    business_value: float

    @property
    def expansion_targets(self) -> List[str]:
        title_token_set = set(self.title_tokens)
        return [
            token
            for token in self.visual_attributes + self.demand_tokens
            if token not in title_token_set
        ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pad_title_tokens(tokens: List[str]) -> torch.Tensor:
    ids = [TOKEN_TO_ID[token] for token in tokens[:MAX_TITLE_LEN]]
    ids += [PAD_ID] * (MAX_TITLE_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_image_attributes(attributes: List[str]) -> torch.Tensor:
    feature = torch.zeros(IMAGE_DIM, dtype=torch.float32)
    for token in attributes:
        feature[VISUAL_ATTRIBUTES.index(token)] = 1.0
    return feature


def reward_vector(product: Product) -> torch.Tensor:
    target = torch.zeros(len(EXPANSION_VOCAB), dtype=torch.float32)
    for token in product.expansion_targets:
        reward = 1.0
        if token in product.visual_attributes:
            reward += 0.7
        reward += COMMERCIAL_WEIGHTS.get(token, 0.2)
        target[EXPANSION_TO_ID[token]] = reward
    if target.sum() == 0:
        target[EXPANSION_TO_ID[random.choice(EXPANSION_VOCAB)]] = 1.0
    return target


def maybe_mask_title(product: Product, rng: random.Random) -> List[str]:
    tokens = list(product.title_tokens)
    hidden_visual = [token for token in product.visual_attributes if token in tokens]
    if hidden_visual and rng.random() < 0.7:
        token = rng.choice(hidden_visual)
        tokens.remove(token)
    return tokens


def random_product(rng: random.Random) -> Product:
    category = rng.choice(CATEGORIES)
    style = rng.choice(VISIBLE_STYLES)
    shown_visual = rng.choice(VISUAL_ATTRIBUTES)
    hidden_visual = rng.sample([token for token in VISUAL_ATTRIBUTES if token != shown_visual], 2)
    demand_tokens = rng.sample(DEMAND_TOKENS, 2)
    title_tokens = [category, style, shown_visual]
    if rng.random() < 0.35:
        title_tokens.append(rng.choice([demand_tokens[0], style]))
    business_value = round(1.0 + rng.random() * 1.8 + sum(COMMERCIAL_WEIGHTS.get(token, 0.2) for token in demand_tokens) * 0.2, 3)
    return Product(
        category=category,
        title_tokens=title_tokens,
        visual_attributes=[shown_visual] + hidden_visual,
        demand_tokens=demand_tokens,
        business_value=business_value,
    )


class SAMD2QDataset(Dataset):
    def __init__(self, split: str, size: int, seed: int):
        self.split = split
        self.size = size
        self.seed = seed
        self.items = self._build_items()

    def _build_items(self) -> List[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        items: List[Dict[str, torch.Tensor]] = []
        while len(items) < self.size:
            product = random_product(rng)
            if not product.expansion_targets:
                continue
            masked_title = maybe_mask_title(product, rng)
            items.append(
                {
                    "title_ids": pad_title_tokens(masked_title),
                    "image_features": encode_image_attributes(product.visual_attributes),
                    "reward_targets": reward_vector(product),
                    "positive_mask": (reward_vector(product) > 0).float(),
                }
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.items[index]


def build_datasets(train_size: int = 720, val_size: int = 180, test_size: int = 240):
    return {
        "train": SAMD2QDataset("train", train_size, seed=11),
        "val": SAMD2QDataset("val", val_size, seed=17),
        "test": SAMD2QDataset("test", test_size, seed=23),
    }


def build_search_benchmark(size: int = 120, seed: int = 29) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    benchmark: List[Dict[str, object]] = []
    for _ in range(size):
        product = random_product(rng)
        if len(product.expansion_targets) < 2:
            continue
        query_terms = [product.category] + product.expansion_targets[:2]
        benchmark.append(
            {
                "product": product,
                "query_terms": query_terms,
            }
        )
    return benchmark
