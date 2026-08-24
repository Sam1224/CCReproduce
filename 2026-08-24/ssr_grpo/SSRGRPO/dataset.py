import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


QUERIES = ["running face mask", "organic milk", "cotton summer shirt", "wireless fast charger", "kids water park"]
ITEMS = [
    "breathable uv running mask for women",
    "organic almond milk carton",
    "large cotton short sleeve shirt",
    "wireless fast charging pad",
    "children family water park ticket",
    "fashion silk scarf",
    "gaming keyboard rgb",
    "protein snack bar",
]
VOCAB = {"pad": 0, "unk": 1}
for text in QUERIES + ITEMS:
    for token in text.split():
        VOCAB.setdefault(token, len(VOCAB))


class RetrievalToyDataset(Dataset):
    def __init__(self, size: int = 256, seed: int = 13):
        random.seed(seed)
        self.rows: List[Dict[str, int]] = []
        positives = [0, 1, 2, 3, 4]
        for _ in range(size):
            query_index = random.randrange(len(QUERIES))
            positive_index = positives[query_index]
            negatives = [idx for idx in range(len(ITEMS)) if idx != positive_index]
            hard_negative = random.choice(negatives)
            self.rows.append({"query": query_index, "positive": positive_index, "negative": hard_negative})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "query_tokens": encode(QUERIES[row["query"]]),
            "positive_tokens": encode(ITEMS[row["positive"]]),
            "negative_tokens": encode(ITEMS[row["negative"]]),
        }


def encode(text: str, max_length: int = 10) -> torch.Tensor:
    ids = [VOCAB.get(token, VOCAB["unk"]) for token in text.split()]
    ids = ids[:max_length] + [0] * max(0, max_length - len(ids))
    return torch.tensor(ids, dtype=torch.long)
