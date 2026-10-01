from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


TAGS = ["normal", "spam", "misleading", "restricted", "quality_issue"]
TEXT_DIM = 12
IMAGE_DIM = 10
META_DIM = 6
FEATURE_DIM = TEXT_DIM + IMAGE_DIM + META_DIM


@dataclass
class GovernanceBatch:
    features: torch.Tensor
    labels: torch.Tensor
    group_ids: torch.Tensor


@dataclass
class GovernanceSplits:
    train: GovernanceBatch
    valid: GovernanceBatch
    test: GovernanceBatch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _make_split(n: int, seed: int, positive_rate: float = 0.12) -> GovernanceBatch:
    generator = torch.Generator().manual_seed(seed)
    labels = (torch.rand(n, generator=generator) < positive_rate).long()
    text = torch.randn(n, TEXT_DIM, generator=generator)
    image = torch.randn(n, IMAGE_DIM, generator=generator)
    meta = torch.randn(n, META_DIM, generator=generator) * 0.7

    # Positive samples are intentionally rare but carry weak multimodal signals.
    pos = labels.float().unsqueeze(1)
    text[:, :4] += pos * torch.tensor([1.4, 1.0, 0.7, 0.4])
    image[:, :3] += pos * torch.tensor([1.2, 0.8, 0.5])
    meta[:, :2] += pos * torch.tensor([0.9, 0.6])

    # A majority-class shortcut: some normal samples have popular-creator metadata
    # that can inflate accuracy while hurting violation ranking.
    shortcut = ((labels == 0) & (torch.rand(n, generator=generator) < 0.35)).float().unsqueeze(1)
    meta[:, 2:4] += shortcut * torch.tensor([1.8, 1.2])

    features = torch.cat([text, image, meta], dim=1).float()
    group_ids = torch.randint(0, 8, (n,), generator=generator)
    return GovernanceBatch(features=features, labels=labels, group_ids=group_ids)


def build_splits(seed: int = 13) -> GovernanceSplits:
    return GovernanceSplits(
        train=_make_split(720, seed=seed, positive_rate=0.12),
        valid=_make_split(260, seed=seed + 1, positive_rate=0.12),
        test=_make_split(260, seed=seed + 2, positive_rate=0.12),
    )


class GovernanceDataset(Dataset):
    def __init__(self, batch: GovernanceBatch):
        self.batch = batch

    def __len__(self) -> int:
        return int(self.batch.labels.numel())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.batch.features[idx],
            "label": self.batch.labels[idx],
            "group_id": self.batch.group_ids[idx],
        }


def positive_negative_pairs(labels: torch.Tensor, max_pairs: int = 4096, seed: int = 0) -> List[Tuple[int, int]]:
    positives = torch.where(labels == 1)[0].tolist()
    negatives = torch.where(labels == 0)[0].tolist()
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    for pos_idx in positives:
        sampled = rng.sample(negatives, k=min(24, len(negatives)))
        for neg_idx in sampled:
            pairs.append((pos_idx, neg_idx))
    rng.shuffle(pairs)
    return pairs[:max_pairs]
