import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

CATEGORIES = ["normal", "counterfeit", "misleading", "prohibited", "medical", "adult"]
BASE_TOKENS = {
    "<pad>": 0,
    "authentic": 1,
    "discount": 2,
    "replica": 3,
    "miracle": 4,
    "medicine": 5,
    "restricted": 6,
    "adult": 7,
    "safe": 8,
    "creator": 9,
    "live": 10,
    "shop": 11,
    "brand": 12,
    "claim": 13,
    "hidden": 14,
    "split": 15,
    "homoglyph": 16,
    "emoji": 17,
}
VIOLATION_TOKENS = {
    "counterfeit": ["replica", "brand"],
    "misleading": ["miracle", "claim"],
    "prohibited": ["restricted", "shop"],
    "medical": ["medicine", "claim"],
    "adult": ["adult", "hidden"],
}


@dataclass
class EvadeExample:
    token_ids: torch.Tensor
    image: torch.Tensor
    label: int
    violation: float
    evasion_mask: torch.Tensor
    consistency: float


def tokenize(tokens: List[str], max_len: int = 12) -> torch.Tensor:
    token_ids = [BASE_TOKENS.get(token, 0) for token in tokens[:max_len]]
    token_ids += [0] * max(0, max_len - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)


def make_patch(image: torch.Tensor, row: int, col: int, color: torch.Tensor, size: int = 5) -> torch.Tensor:
    image[:, row : row + size, col : col + size] = color.view(3, 1, 1)
    return image


class ToyEvadeBenchDataset(Dataset):
    def __init__(self, n: int = 512, image_size: int = 24, seed: int = 18):
        self.n = n
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.samples = [self._make_one() for _ in range(n)]

    def _image_for_category(self, category: str, evasive: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.ones(3, self.image_size, self.image_size) * 0.93
        mask = torch.zeros(1, self.image_size, self.image_size)
        colors = {
            "normal": torch.tensor([0.2, 0.7, 0.4]),
            "counterfeit": torch.tensor([0.9, 0.1, 0.1]),
            "misleading": torch.tensor([0.9, 0.55, 0.05]),
            "prohibited": torch.tensor([0.55, 0.1, 0.75]),
            "medical": torch.tensor([0.1, 0.45, 0.95]),
            "adult": torch.tensor([0.95, 0.25, 0.55]),
        }
        row = self.rng.choice([2, 9, 16])
        col = self.rng.choice([2, 9, 16])
        image = make_patch(image, row, col, colors[category])
        if evasive:
            mask[:, row : row + 5, col : col + 5] = 1.0
            stripe_color = torch.tensor([0.05, 0.05, 0.05])
            for offset in range(0, 5, 2):
                image[:, row + offset : row + offset + 1, col : col + 5] = stripe_color.view(3, 1, 1)
        return image, mask

    def _make_text(self, category: str, evasive: bool) -> List[str]:
        tokens = ["creator", "live", "shop", "discount"]
        if category == "normal":
            tokens += ["authentic", "safe"]
        else:
            tokens += VIOLATION_TOKENS[category]
        if evasive:
            tokens += self.rng.sample(["hidden", "split", "homoglyph", "emoji"], 2)
        self.rng.shuffle(tokens)
        return tokens

    def _make_one(self) -> EvadeExample:
        category = self.rng.choices(CATEGORIES, weights=[0.35, 0.16, 0.16, 0.13, 0.10, 0.10])[0]
        evasive = category != "normal" and self.rng.random() < 0.7
        text_category = category
        if evasive and self.rng.random() < 0.35:
            text_category = "normal"
        image, evasion_mask = self._image_for_category(category, evasive)
        tokens = self._make_text(text_category, evasive)
        consistency = 1.0 if text_category == category else 0.0
        return EvadeExample(
            token_ids=tokenize(tokens),
            image=image,
            label=CATEGORIES.index(category),
            violation=0.0 if category == "normal" else 1.0,
            evasion_mask=evasion_mask,
            consistency=consistency,
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "token_ids": sample.token_ids,
            "image": sample.image,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "violation": torch.tensor([sample.violation], dtype=torch.float32),
            "evasion_mask": sample.evasion_mask,
            "consistency": torch.tensor([sample.consistency], dtype=torch.float32),
        }


def make_loaders(batch_size: int = 32):
    train = ToyEvadeBenchDataset(800, seed=20260718)
    test = ToyEvadeBenchDataset(200, seed=20260719)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)
