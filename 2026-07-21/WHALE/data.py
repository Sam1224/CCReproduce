import random
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class WhaleExample:
    user_id: int
    item_id: int
    context_id: int
    user_pref: int
    item_category: int
    focus_category: int
    sequence: torch.Tensor
    label: float


class WhaleToyDataset(Dataset):
    def __init__(self, n: int = 4096, seq_len: int = 20, num_users: int = 128, num_items: int = 96, num_contexts: int = 8, seed: int = 7):
        self.rng = random.Random(seed)
        self.seq_len = seq_len
        self.num_users = num_users
        self.num_items = num_items
        self.num_contexts = num_contexts
        self.num_categories = 12
        self.user_pref = [self.rng.randrange(self.num_categories) for _ in range(num_users)]
        self.item_cat = [self.rng.randrange(self.num_categories) for _ in range(num_items)]
        self.samples = [self._make_one() for _ in range(n)]

    def _make_one(self) -> WhaleExample:
        user_id = self.rng.randrange(self.num_users)
        context_id = self.rng.randrange(self.num_contexts)
        pref = self.user_pref[user_id]
        focus_cat = (pref + context_id) % self.num_categories
        sequence_items = []
        for idx in range(self.seq_len):
            if idx > self.seq_len - 6 and self.rng.random() < 0.88:
                target_cat = focus_cat
            elif self.rng.random() < 0.5:
                target_cat = pref
            else:
                target_cat = self.rng.randrange(self.num_categories)
            candidates = [i for i, c in enumerate(self.item_cat) if c == target_cat]
            sequence_items.append(self.rng.choice(candidates))
        recent_match = self.rng.random() < 0.5
        label = 1.0 if recent_match else 0.0
        target_cat = focus_cat if recent_match else (focus_cat + self.rng.randrange(3, self.num_categories)) % self.num_categories
        item_id = self.rng.choice([i for i, c in enumerate(self.item_cat) if c == target_cat])
        return WhaleExample(
            user_id=user_id,
            item_id=item_id,
            context_id=context_id,
            user_pref=pref,
            item_category=self.item_cat[item_id],
            focus_category=focus_cat,
            sequence=torch.tensor(sequence_items, dtype=torch.long),
            label=label,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.samples[idx]
        return {
            "user_id": torch.tensor(ex.user_id, dtype=torch.long),
            "item_id": torch.tensor(ex.item_id, dtype=torch.long),
            "context_id": torch.tensor(ex.context_id, dtype=torch.long),
            "user_pref": torch.tensor(ex.user_pref, dtype=torch.long),
            "item_category": torch.tensor(ex.item_category, dtype=torch.long),
            "focus_category": torch.tensor(ex.focus_category, dtype=torch.long),
            "sequence": ex.sequence,
            "label": torch.tensor([ex.label], dtype=torch.float32),
        }


def make_loaders(batch_size: int = 64):
    train = WhaleToyDataset(n=4096, seed=11)
    test = WhaleToyDataset(n=1024, seed=23)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
    )
