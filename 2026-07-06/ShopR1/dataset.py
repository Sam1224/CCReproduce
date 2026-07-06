import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


ACTIONS = ["search", "click", "filter", "add_to_cart", "purchase", "abandon"]
ATTRIBUTES = ["query", "product", "price", "rating", "review", "shipping", "reason"]
VALUES = ["cheap", "premium", "fast", "safe", "popular", "uncertain", "none"]


TOY_SAMPLES = [
    {
        "context": "user wants wireless earbuds under 50 dollars with good battery and reads many reviews",
        "rationale": "price constraint and review evidence dominate the next step",
        "action_type": "filter",
        "attribute": "price",
        "value": "cheap",
        "difficulty": 1.4,
    },
    {
        "context": "user searched summer dress clicks a floral product with high rating and free shipping",
        "rationale": "the clicked product matches style rating and shipping preference",
        "action_type": "add_to_cart",
        "attribute": "product",
        "value": "popular",
        "difficulty": 1.0,
    },
    {
        "context": "user compares baby bottle sterilizer reviews and notices safety complaints",
        "rationale": "negative safety reviews make the shopper stop the purchase",
        "action_type": "abandon",
        "attribute": "review",
        "value": "safe",
        "difficulty": 1.8,
    },
    {
        "context": "user wants running shoes from a known brand and finds a premium listing",
        "rationale": "brand match and premium intent make product click likely",
        "action_type": "click",
        "attribute": "product",
        "value": "premium",
        "difficulty": 1.1,
    },
    {
        "context": "user enters waterproof phone pouch beach vacation and scans low price items",
        "rationale": "the query should be refined before selecting a product",
        "action_type": "search",
        "attribute": "query",
        "value": "cheap",
        "difficulty": 1.2,
    },
    {
        "context": "user has compared office chair reviews price and shipping and returns to cart",
        "rationale": "enough evidence has been collected to complete checkout",
        "action_type": "purchase",
        "attribute": "shipping",
        "value": "fast",
        "difficulty": 1.5,
    },
]


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    @classmethod
    def build(cls, texts: List[str], min_freq: int = 1) -> "Vocabulary":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in text.lower().split():
                counts[token] = counts.get(token, 0) + 1
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, count in sorted(counts.items()):
            if count >= min_freq:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.token_to_id.get(token, 1) for token in text.lower().split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class ShoppingBehaviorDataset(Dataset):
    def __init__(self, path: str = "", max_len: int = 32, vocab: Vocabulary | None = None):
        self.max_len = max_len
        self.samples = self._load_samples(path)
        texts = [sample["context"] + " " + sample["rationale"] for sample in self.samples]
        self.vocab = vocab or Vocabulary.build(texts)

    def _load_samples(self, path: str) -> List[dict]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return TOY_SAMPLES

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": self.vocab.encode(sample["context"], self.max_len),
            "rationale_ids": self.vocab.encode(sample["rationale"], self.max_len),
            "action_type": torch.tensor(ACTIONS.index(sample["action_type"]), dtype=torch.long),
            "attribute": torch.tensor(ATTRIBUTES.index(sample["attribute"]), dtype=torch.long),
            "value": torch.tensor(VALUES.index(sample["value"]), dtype=torch.long),
            "difficulty": torch.tensor(float(sample.get("difficulty", 1.0)), dtype=torch.float),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def save_toy_dataset(path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(TOY_SAMPLES, handle, ensure_ascii=False, indent=2)
