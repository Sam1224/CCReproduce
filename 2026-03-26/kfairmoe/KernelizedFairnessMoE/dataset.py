from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class FairRecSample:
    x: torch.Tensor  # (L, d_model)
    y: torch.Tensor  # (L,) next-item ids
    sensitive: torch.Tensor  # (L,) sensitive attribute ids


class FairRecToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        d_model: int = 128,
        num_items: int = 200,
        num_sensitive: int = 2,
        min_len: int = 8,
        max_len: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.d_model = d_model
        self.num_items = num_items
        self.num_sensitive = num_sensitive
        self.min_len = min_len
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        length = self.rng.randint(self.min_len, self.max_len)

        # Toy hidden states from a base LLM.
        x = torch.randn(length, self.d_model)

        # Recommendation labels (per token / per position).
        y = torch.randint(low=0, high=self.num_items, size=(length,))

        # Sensitive attribute (correlated with y to make leakage measurable).
        sensitive = (y % self.num_sensitive).clone()

        return {"x": x, "y": y, "sensitive": sensitive}


def collate_fairrec(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([b["x"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    d_model = batch[0]["x"].shape[-1]

    x = torch.zeros(len(batch), max_len, d_model)
    y = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    sensitive = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        l = sample["x"].shape[0]
        x[i, :l] = sample["x"]
        y[i, :l] = sample["y"]
        sensitive[i, :l] = sample["sensitive"]
        attn_mask[i, :l] = True

    return {"x": x, "y": y, "sensitive": sensitive, "attn_mask": attn_mask, "lengths": lengths}
