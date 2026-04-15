from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticDataConfig:
    num_instances: int = 8000
    max_steps: int = 32
    feature_dim: int = 16
    domain_shift: float = 0.0
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic(cfg: SyntheticDataConfig) -> dict:
    """Generate synthetic reasoning trajectories.

    For each instance i and step t:
    - x[i,t] is a feature vector.
    - y[i,t] is whether stopping at step t yields a correct answer.
    - c[i,t] is a noisy pseudo-label (standing in for self-consistency/supervised signals).

    We model the probability of correctness to generally increase with t, with per-instance
    difficulty and an optional domain shift.
    """

    rng = np.random.default_rng(cfg.seed)

    # Instance difficulty: higher => harder
    difficulty = rng.normal(loc=0.0 + cfg.domain_shift, scale=0.8, size=(cfg.num_instances, 1))

    # Step progress: increases with t
    t = np.arange(cfg.max_steps, dtype=np.float32)[None, :]  # [1, T]
    progress = (t / (cfg.max_steps - 1)) * 12.0 - 6.0  # roughly [-6,6]

    # Base correctness probability
    logits = progress - difficulty  # [N, T]
    p_correct = _sigmoid(logits)

    # Sample true correctness if stop at step t
    y = rng.binomial(1, p_correct).astype(np.float32)

    # Feature construction: correlated with p_correct + noise
    # x[...,0] ~ progress, x[...,1] ~ difficulty, others random
    x = rng.normal(size=(cfg.num_instances, cfg.max_steps, cfg.feature_dim)).astype(np.float32)
    x[..., 0] = progress
    x[..., 1] = -difficulty
    noise_scale = 0.05 + 0.20 * abs(cfg.domain_shift)
    x[..., 2] = np.clip(
        p_correct + rng.normal(scale=noise_scale, size=p_correct.shape).astype(np.float32),
        0.0,
        1.0,
    )

    # Pseudo-labels: noisy versions of y (e.g., self-consistency)
    flip = rng.binomial(1, 0.10, size=y.shape).astype(np.float32)
    c = np.abs(y - flip)  # flip 15%

    mask = np.ones((cfg.num_instances, cfg.max_steps), dtype=np.float32)

    return {
        "x": x,
        "y": y,
        "c": c,
        "mask": mask,
    }


class SyntheticReasoningDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray, mask: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.c = torch.from_numpy(c)
        self.mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "c": self.c[idx],
            "mask": self.mask[idx],
        }


def make_splits(
    cfg: SyntheticDataConfig,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
) -> Tuple[SyntheticReasoningDataset, SyntheticReasoningDataset, SyntheticReasoningDataset]:
    data = generate_synthetic(cfg)
    n = data["x"].shape[0]
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)

    train = SyntheticReasoningDataset(
        data["x"][:n_train], data["y"][:n_train], data["c"][:n_train], data["mask"][:n_train]
    )
    cal = SyntheticReasoningDataset(
        data["x"][n_train : n_train + n_cal],
        data["y"][n_train : n_train + n_cal],
        data["c"][n_train : n_train + n_cal],
        data["mask"][n_train : n_train + n_cal],
    )
    test = SyntheticReasoningDataset(
        data["x"][n_train + n_cal :],
        data["y"][n_train + n_cal :],
        data["c"][n_train + n_cal :],
        data["mask"][n_train + n_cal :],
    )
    return train, cal, test
