import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


VOCAB = {
    "pad": 0,
    "organic": 1,
    "gluten": 2,
    "free": 3,
    "almond": 4,
    "milk": 5,
    "wireless": 6,
    "charger": 7,
    "cotton": 8,
    "shirt": 9,
    "brand": 10,
    "size": 11,
    "diet": 12,
    "material": 13,
    "image": 14,
    "merchant": 15,
    "web": 16,
    "unknown": 17,
}

ATTRIBUTES = ["diet", "brand", "material", "size"]
VALUES = ["unknown", "organic", "gluten_free", "almond", "wireless", "cotton", "small", "large"]
VERDICTS = ["write", "review", "block"]


@dataclass
class CatalogExample:
    sku_text: str
    merchant_evidence: str
    web_evidence: str
    image_evidence: str
    attribute: str
    value: str
    verdict: str


def tokenize(text: str, max_length: int = 20) -> torch.Tensor:
    tokens = [VOCAB.get(piece.lower().replace("-", "_"), VOCAB["unknown"]) for piece in text.split()]
    tokens = tokens[:max_length] + [VOCAB["pad"]] * max(0, max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


class ToyCatalogDataset(Dataset):
    def __init__(self, size: int = 256, seed: int = 7):
        random.seed(seed)
        self.examples: List[CatalogExample] = []
        templates: List[Tuple[str, str, str, str, str, str]] = [
            ("organic almond milk", "merchant organic milk", "web organic almond brand", "image carton milk", "diet", "organic"),
            ("gluten free snack", "merchant gluten free", "web dietary tag gluten free", "image snack label", "diet", "gluten_free"),
            ("wireless charger", "merchant electronics charger", "web brand wireless", "image charger pad", "brand", "wireless"),
            ("cotton shirt large", "merchant cotton apparel", "web material cotton", "image shirt label", "material", "cotton"),
        ]
        for index in range(size):
            sku, merchant, web, image, attribute, value = random.choice(templates)
            noisy = random.random() < 0.18
            candidate = random.choice(VALUES[1:]) if noisy else value
            verdict = "write" if candidate == value else random.choice(["review", "block"])
            self.examples.append(CatalogExample(sku, merchant, web, image, attribute, candidate, verdict))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "sku_tokens": tokenize(example.sku_text),
            "merchant_tokens": tokenize(example.merchant_evidence),
            "web_tokens": tokenize(example.web_evidence),
            "image_tokens": tokenize(example.image_evidence),
            "attribute": torch.tensor(ATTRIBUTES.index(example.attribute), dtype=torch.long),
            "value": torch.tensor(VALUES.index(example.value), dtype=torch.long),
            "verdict": torch.tensor(VERDICTS.index(example.verdict), dtype=torch.long),
        }
