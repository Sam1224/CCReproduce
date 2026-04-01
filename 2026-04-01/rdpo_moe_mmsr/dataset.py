from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Pair:
    seq_item_ids: torch.Tensor  # [L]
    seq_modality_ids: torch.Tensor  # [L]
    pos_next: torch.Tensor
    neg_next: torch.Tensor


class ToyMMSRDPODataset(Dataset):
    """Toy multimodal sequential recommendation pairs for DPO-style training.

    Paper: Aligning Multimodal Sequential Recommendations via Robust Direct Preference Optimization with Sparse MoE
    (arXiv:2603.29259)

    We simulate implicit feedback by generating (preferred, rejected) next-item pairs.
    """

    def __init__(
        self,
        n: int = 30000,
        *,
        num_items: int = 8000,
        num_modalities: int = 3,
        seq_len: int = 20,
        num_categories: int = 80,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.num_items = num_items
        self.num_modalities = num_modalities
        self.seq_len = seq_len
        self.num_categories = num_categories

        rng = random.Random(seed)
        self._rows = []
        for _ in range(n):
            # Build a sequence with a dominant category.
            dom = rng.randrange(num_categories)
            seq = []
            mod = []
            for _t in range(seq_len):
                cat = dom if rng.random() < 0.6 else rng.randrange(num_categories)
                seq.append(self._item_for_cat(cat, rng))
                mod.append(rng.randrange(num_modalities))

            pos = self._item_for_cat(dom, rng)
            neg_cat = rng.randrange(num_categories)
            while neg_cat == dom:
                neg_cat = rng.randrange(num_categories)
            neg = self._item_for_cat(neg_cat, rng)
            self._rows.append((seq, mod, pos, neg))

    def _item_for_cat(self, cat: int, rng: random.Random) -> int:
        block = self.num_items // self.num_categories
        start = cat * block
        end = min(self.num_items, start + block)
        if end <= start:
            return rng.randrange(self.num_items)
        return rng.randrange(start, end)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Pair:
        seq, mod, pos, neg = self._rows[idx]
        return Pair(
            seq_item_ids=torch.tensor(seq, dtype=torch.long),
            seq_modality_ids=torch.tensor(mod, dtype=torch.long),
            pos_next=torch.tensor(pos, dtype=torch.long),
            neg_next=torch.tensor(neg, dtype=torch.long),
        )
