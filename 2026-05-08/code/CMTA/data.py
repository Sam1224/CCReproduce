"""Toy synthetic dataset for CMTA reproduction.

Each sample = (clip_features, caption_features, label).
- clip_features: [T, D] per-frame visual feature
- caption_features: [T, D] per-frame text feature (BLIP caption -> CLIP text)
- label: 0 = real, 1 = AIGV

To mimic the paper's central observation, AIGV samples have *unnaturally stable*
cross-modal cosine alignment over time, while real samples wobble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyVideoConfig:
    n_train: int = 1024
    n_eval: int = 256
    seq_len: int = 16
    feature_dim: int = 64
    seed: int = 7


class ToyVideoDataset(Dataset):
    def __init__(self, cfg: ToyVideoConfig, split: str = "train") -> None:
        rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))
        n = cfg.n_train if split == "train" else cfg.n_eval
        v = rng.standard_normal((n, cfg.seq_len, cfg.feature_dim)).astype(np.float32)
        c = v.copy()
        labels = rng.integers(0, 2, size=n).astype(np.int64)
        # Real videos: caption features fluctuate around frame features (semantic drift).
        # AIGV videos: caption features track frame features tightly (stable alignment).
        for i in range(n):
            if labels[i] == 0:  # real -> add per-frame drift
                drift = rng.standard_normal((cfg.seq_len, cfg.feature_dim)).astype(np.float32) * 0.8
                c[i] = c[i] + drift
            else:  # AIGV -> add small isotropic noise
                c[i] = c[i] + rng.standard_normal((cfg.seq_len, cfg.feature_dim)).astype(np.float32) * 0.05
        self.v, self.c, self.y = v, c, labels
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "v": torch.from_numpy(self.v[idx]),
            "c": torch.from_numpy(self.c[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }
