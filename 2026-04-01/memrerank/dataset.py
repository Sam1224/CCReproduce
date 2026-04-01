from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairBatch:
    history_item_ids: torch.Tensor  # [H]
    query_token_ids: torch.Tensor  # [T]
    pos_item_id: torch.Tensor
    neg_item_id: torch.Tensor


class ToyMemRerankDataset(Dataset):
    """Toy dataset for preference-memory reranking.

    We simulate:
    - a user with a short purchase/browse history
    - a query (text tokens)
    - a preferred (positive) item and a rejected (negative) item

    Positives share the dominant category in the history; negatives are from other categories.
    """

    def __init__(
        self,
        n: int = 20000,
        *,
        num_items: int = 5000,
        num_categories: int = 50,
        history_len: int = 20,
        query_len: int = 16,
        vocab_size: int = 2048,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.num_items = num_items
        self.num_categories = num_categories
        self.history_len = history_len
        self.query_len = query_len
        self.vocab_size = vocab_size

        rng = random.Random(seed)
        self._user_pref_cat = [rng.randrange(num_categories) for _ in range(n)]
        self._histories = []
        self._queries = []
        self._pos = []
        self._neg = []

        for i in range(n):
            pref = self._user_pref_cat[i]
            history = []
            for _ in range(history_len):
                cat = pref if rng.random() < 0.7 else rng.randrange(num_categories)
                history.append(self._item_for_cat(cat, rng))
            self._histories.append(history)

            self._queries.append([rng.randrange(vocab_size) for _ in range(query_len)])

            self._pos.append(self._item_for_cat(pref, rng))
            neg_cat = rng.randrange(num_categories)
            while neg_cat == pref:
                neg_cat = rng.randrange(num_categories)
            self._neg.append(self._item_for_cat(neg_cat, rng))

    def _item_for_cat(self, cat: int, rng: random.Random) -> int:
        # Assign each category a contiguous block of item IDs.
        block = self.num_items // self.num_categories
        start = cat * block
        end = min(self.num_items, start + block)
        if end <= start:
            return rng.randrange(self.num_items)
        return rng.randrange(start, end)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> PairBatch:
        return PairBatch(
            history_item_ids=torch.tensor(self._histories[idx], dtype=torch.long),
            query_token_ids=torch.tensor(self._queries[idx], dtype=torch.long),
            pos_item_id=torch.tensor(self._pos[idx], dtype=torch.long),
            neg_item_id=torch.tensor(self._neg[idx], dtype=torch.long),
        )
