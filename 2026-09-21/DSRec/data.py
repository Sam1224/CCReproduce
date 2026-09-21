from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    items: torch.Tensor
    deltas: torch.Tensor
    targets: torch.Tensor


class SyntheticProductSequenceDataset(Dataset):
    def __init__(self, size: int = 512, seq_len: int = 12, num_items: int = 80, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        self.items = torch.randint(1, num_items, (size, seq_len), generator=generator)
        self.deltas = torch.rand(size, seq_len, generator=generator) * 6.0
        long_signal = torch.mode(self.items[:, : seq_len // 2], dim=1).values
        short_signal = self.items[:, -1]
        switch = (self.deltas[:, -1] < 2.5).long()
        self.targets = torch.where(switch.bool(), short_signal, long_signal)

    def __len__(self) -> int:
        return self.items.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "items": self.items[index],
            "deltas": self.deltas[index],
            "targets": self.targets[index],
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> Batch:
    return Batch(
        items=torch.stack([row["items"] for row in batch]),
        deltas=torch.stack([row["deltas"] for row in batch]),
        targets=torch.stack([row["targets"] for row in batch]),
    )
