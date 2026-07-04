import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


CATEGORIES = [
    "apple",
    "lettuce",
    "milk",
    "bread",
    "coffee",
    "diapers",
    "paper towel",
    "yogurt",
    "banana",
    "eggs",
]

QUERIES = [
    "breakfast staples",
    "weekly fresh produce",
    "baby household refill",
    "office snack restock",
    "healthy lunch",
]

VOCAB = ["<pad>", "<bos>", "<eos>"] + CATEGORIES + QUERIES
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}


@dataclass
class GroceryExample:
    history: List[int]
    query: int
    target: List[int]
    rebuy_prior: List[float]


class GroceryDataset(Dataset):
    def __init__(self, size: int = 320, seed: int = 7):
        self.examples = make_examples(size, seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return {
            "history": torch.tensor(example.history, dtype=torch.long),
            "query": torch.tensor(example.query, dtype=torch.long),
            "target": torch.tensor(example.target, dtype=torch.float32),
            "rebuy_prior": torch.tensor(example.rebuy_prior, dtype=torch.float32),
        }


def make_examples(size: int, seed: int) -> List[GroceryExample]:
    random.seed(seed)
    examples: List[GroceryExample] = []
    for _ in range(size):
        base = random.sample(range(len(CATEGORIES)), k=4)
        history = [random.choice(base) for _ in range(12)]
        query_index = random.randrange(len(QUERIES))
        target_ids = set(random.sample(base, k=2))
        if "fresh" in QUERIES[query_index] or "healthy" in QUERIES[query_index]:
            target_ids.update([0, 1, 8])
        if "baby" in QUERIES[query_index]:
            target_ids.update([5, 6])
        target = [1.0 if category_index in target_ids else 0.0 for category_index in range(len(CATEGORIES))]
        counts: Dict[int, int] = {category_index: history.count(category_index) for category_index in range(len(CATEGORIES))}
        total = max(1, len(history))
        rebuy_prior = [counts[index] / total for index in range(len(CATEGORIES))]
        examples.append(
            GroceryExample(
                history=history,
                query=len(CATEGORIES) + query_index,
                target=target,
                rebuy_prior=rebuy_prior,
            )
        )
    return examples


def collate(batch: List[dict]) -> dict:
    return {
        "history": torch.stack([item["history"] for item in batch]),
        "query": torch.stack([item["query"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "rebuy_prior": torch.stack([item["rebuy_prior"] for item in batch]),
    }


def category_trie() -> Dict:
    root: Dict = {}
    for category in CATEGORIES:
        node = root
        for token in category.split():
            node = node.setdefault(token, {})
        node["<eos>"] = {}
    return root
