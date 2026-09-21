from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class UnitStats:
    name: str
    support: int
    accepted: int
    correct: int


class SelectivePredictionDataset(Dataset):
    def __init__(self, size: int = 600, groups: int = 12, seed: int = 41):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(size, 6, generator=generator)
        self.group = torch.randint(0, groups, (size,), generator=generator)
        latent = self.features[:, 0] + 0.45 * self.features[:, 1] - 0.08 * self.group.float()
        self.label = (latent > 0).long()
        noise = 0.12 + 0.04 * self.group.float()
        score = torch.sigmoid(latent.abs() + torch.randn(size, generator=generator) * noise)
        self.confidence = score.clamp(0, 1)

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.features[idx], "y": self.label[idx], "group": self.group[idx], "confidence": self.confidence[idx]}
