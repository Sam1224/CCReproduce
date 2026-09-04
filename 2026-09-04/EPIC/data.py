from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

CATEGORY_NAMES = ["beauty", "fashion", "food", "home", "sports"]
BRAND_NAMES = ["nova", "luma", "ora", "zen", "pulse", "craft"]
STYLE_NAMES = ["budget", "premium", "casual", "pro"]
TOKEN_DIM = 4
CODEBOOK_SIZE = 8


@dataclass(frozen=True)
class CatalogItem:
    item_id: int
    sid: Tuple[int, int, int, int]
    category: int
    brand: int
    style: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_catalog() -> List[CatalogItem]:
    catalog: List[CatalogItem] = []
    item_id = 0
    for category in range(len(CATEGORY_NAMES)):
        for brand in range(len(BRAND_NAMES)):
            for style in range(len(STYLE_NAMES)):
                sid = (category, brand, style, (category + brand + style) % CODEBOOK_SIZE)
                catalog.append(CatalogItem(item_id=item_id, sid=sid, category=category, brand=brand, style=style))
                item_id += 1
    return catalog


CATALOG = build_catalog()
ITEM_ID_TO_ITEM = {item.item_id: item for item in CATALOG}
NUM_ITEMS = len(CATALOG)
NUM_CATEGORIES = len(CATEGORY_NAMES)


def next_item_candidates(last_item: CatalogItem) -> List[CatalogItem]:
    return [item for item in CATALOG if item.category == last_item.category and item.brand == last_item.brand]


class SyntheticSequenceDataset(Dataset):
    def __init__(self, split: str, size: int, seed: int = 42, history_len: int = 6):
        self.split = split
        self.size = size
        self.seed = seed
        self.history_len = history_len
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        samples: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.size):
            user_category = rng.randrange(NUM_CATEGORIES)
            user_brand = rng.randrange(len(BRAND_NAMES))
            history: List[int] = []
            for step in range(self.history_len):
                style = (step + rng.randrange(len(STYLE_NAMES))) % len(STYLE_NAMES)
                sid = (user_category, user_brand, style, (user_category + user_brand + style) % CODEBOOK_SIZE)
                matched = next(item for item in CATALOG if item.sid == sid)
                history.append(matched.item_id)

            target_style = (history[-1] + user_category + user_brand) % len(STYLE_NAMES)
            target_sid = (user_category, user_brand, target_style, (user_category + user_brand + target_style) % CODEBOOK_SIZE)
            target_item = next(item for item in CATALOG if item.sid == target_sid)
            history_categories = [ITEM_ID_TO_ITEM[item_id].category for item_id in history]
            history_brands = [ITEM_ID_TO_ITEM[item_id].brand for item_id in history]
            history_styles = [ITEM_ID_TO_ITEM[item_id].style for item_id in history]
            samples.append(
                {
                    "history_items": torch.tensor(history, dtype=torch.long),
                    "history_categories": torch.tensor(history_categories, dtype=torch.long),
                    "history_brands": torch.tensor(history_brands, dtype=torch.long),
                    "history_styles": torch.tensor(history_styles, dtype=torch.long),
                    "target_item": torch.tensor(target_item.item_id, dtype=torch.long),
                    "target_sid": torch.tensor(target_item.sid, dtype=torch.long),
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


def build_datasets(train_size: int = 640, val_size: int = 160, test_size: int = 200):
    return {
        "train": SyntheticSequenceDataset("train", train_size, seed=5),
        "val": SyntheticSequenceDataset("val", val_size, seed=9),
        "test": SyntheticSequenceDataset("test", test_size, seed=13),
    }
