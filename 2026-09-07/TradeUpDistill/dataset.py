from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class TradeUpSample:
    base_embedding: torch.Tensor
    candidate_embedding: torch.Tensor
    product_type: torch.Tensor
    label: torch.Tensor
    binary_label: torch.Tensor
    rationale_embedding: torch.Tensor


class ToyTradeUpDataset(Dataset):
    """Synthetic product-pair data with directional trade-up structure.

    Labels mirror the paper's four-class supervision:
    0 unrelated, 1 variant/same-tier, 2 invalid downgrade, 3 valid trade-up.
    """

    def __init__(self, num_samples: int = 2048, embedding_dim: int = 768, num_product_types: int = 12, seed: int = 7) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.base = torch.randn(num_samples, embedding_dim, generator=generator)
        self.product_type = torch.randint(0, num_product_types, (num_samples,), generator=generator)
        type_bias = torch.randn(num_product_types, embedding_dim, generator=generator) * 0.25
        quality_axis = torch.randn(num_product_types, embedding_dim, generator=generator)
        quality_axis = quality_axis / quality_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        labels = torch.randint(0, 4, (num_samples,), generator=generator)
        noise = torch.randn(num_samples, embedding_dim, generator=generator) * 0.08
        direction = quality_axis[self.product_type]
        label_strength = torch.tensor([-0.20, 0.05, -0.45, 0.65])[labels].unsqueeze(-1)
        self.candidate = self.base + type_bias[self.product_type] + label_strength * direction + noise
        rationale_anchor = torch.stack([
            -direction.mean(dim=0),
            type_bias[self.product_type].mean(dim=0),
            -0.5 * direction.mean(dim=0),
            direction.mean(dim=0),
        ])
        self.rationales = rationale_anchor[labels] + torch.randn(num_samples, embedding_dim, generator=generator) * 0.05
        self.labels = labels.long()
        self.binary = (labels == 3).float()
        self.num_product_types = num_product_types
        self.embedding_dim = embedding_dim

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "base_embedding": self.base[idx],
            "candidate_embedding": self.candidate[idx],
            "product_type": self.product_type[idx],
            "label": self.labels[idx],
            "binary_label": self.binary[idx],
            "rationale_embedding": self.rationales[idx],
        }


def split_dataset(dataset: Dataset, train_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
    train_size = int(math.floor(len(dataset) * train_ratio))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(13))
