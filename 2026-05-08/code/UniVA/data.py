"""Synthetic ad-recommendation dataset for UniVA toy reproduction.

Each sample emits:
- user_history: sequence of clicked item ids
- target_item: item id to predict
- target_value: scalar eCPM-like reward used by the value head
- item_meta: tensor of item-level value attributes (eCPM bucket id + category margin id)

The interfaces are aligned with `train.py` / `infer.py` so dropping in a real
WeChat-style production dataset only requires implementing the same returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyConfig:
    num_items: int = 1024
    num_ecpm_buckets: int = 8
    num_margin_buckets: int = 4
    history_len: int = 16
    n_train: int = 4096
    n_eval: int = 512
    seed: int = 42


class ToyAdRecDataset(Dataset):
    def __init__(self, cfg: ToyConfig, split: str = "train") -> None:
        rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))
        n = cfg.n_train if split == "train" else cfg.n_eval
        self.cfg = cfg
        self.history = rng.integers(0, cfg.num_items, size=(n, cfg.history_len), dtype=np.int64)
        # Targets correlate weakly with the last 3 history items (toy signal).
        bias = self.history[:, -3:].sum(axis=1) % cfg.num_items
        noise = rng.integers(0, cfg.num_items, size=n, dtype=np.int64)
        mix = (bias + noise) % cfg.num_items
        self.target = mix.astype(np.int64)
        # Per-item value attributes (shared across split).
        self.item_ecpm_bucket = rng.integers(0, cfg.num_ecpm_buckets, size=cfg.num_items)
        self.item_margin_bucket = rng.integers(0, cfg.num_margin_buckets, size=cfg.num_items)
        # Reward = item ecpm_bucket scaled to [0, 1] -- toy proxy of true eCPM.
        self.reward = self.item_ecpm_bucket[self.target].astype(np.float32) / max(cfg.num_ecpm_buckets - 1, 1)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history": torch.from_numpy(self.history[idx]),
            "target": torch.tensor(self.target[idx], dtype=torch.long),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32),
        }

    def item_value_meta(self) -> torch.Tensor:
        """Tensor of shape [num_items, 2]: (ecpm_bucket, margin_bucket)."""
        return torch.from_numpy(
            np.stack([self.item_ecpm_bucket, self.item_margin_bucket], axis=1).astype(np.int64)
        )
