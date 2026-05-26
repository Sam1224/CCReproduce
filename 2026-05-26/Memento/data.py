from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class ToyAdsConfig:
    num_users: int = 200
    num_ads: int = 500
    num_samples: int = 10_000
    history_len: int = 200
    dim: int = 64
    seed: int = 42


class ToyAdsDataset(Dataset):
    """Synthetic long-history ads dataset.

    Each sample:
    - query_vec: current ad request embedding (D)
    - history_vecs: user history interaction embeddings (H, D)
    - label: click label (0/1)

    We simulate that user preference is stable but slowly drifts; old history can still matter.
    """

    def __init__(self, cfg: ToyAdsConfig, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.cfg = cfg

        set_seed(cfg.seed)

        # Latent user and ad factors
        user_vec = np.random.normal(size=(cfg.num_users, cfg.dim)).astype(np.float32)
        ad_vec = np.random.normal(size=(cfg.num_ads, cfg.dim)).astype(np.float32)

        # Pre-generate per-user long histories
        history_ads = np.random.randint(0, cfg.num_ads, size=(cfg.num_users, cfg.history_len))
        history_noise = np.random.normal(scale=0.1, size=(cfg.num_users, cfg.history_len, cfg.dim)).astype(np.float32)
        history_vecs = ad_vec[history_ads] + history_noise

        # Generate samples
        user_ids = np.random.randint(0, cfg.num_users, size=(cfg.num_samples,))
        ad_ids = np.random.randint(0, cfg.num_ads, size=(cfg.num_samples,))

        query_noise = np.random.normal(scale=0.1, size=(cfg.num_samples, cfg.dim)).astype(np.float32)
        query_vecs = ad_vec[ad_ids] + query_noise

        # Click probability depends on user-ad affinity + a weak long-history effect
        affinity = (user_vec[user_ids] * ad_vec[ad_ids]).sum(axis=1)

        # History effect: similarity of query to a few latent "topics" seen recently
        recent_hist = history_vecs[user_ids, -20:, :]  # (N, 20, D)
        hist_sim = (recent_hist * query_vecs[:, None, :]).sum(axis=-1).max(axis=1)

        logit = 0.8 * affinity / cfg.dim + 0.2 * hist_sim / cfg.dim
        prob = 1 / (1 + np.exp(-logit))
        labels = (np.random.rand(cfg.num_samples) < prob).astype(np.float32)

        # Split
        idx = np.arange(cfg.num_samples)
        np.random.shuffle(idx)
        n_train = int(cfg.num_samples * 0.8)
        n_val = int(cfg.num_samples * 0.1)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        if split == "train":
            sel = train_idx
        elif split == "val":
            sel = val_idx
        else:
            sel = test_idx

        self.user_ids = user_ids[sel]
        self.query_vecs = query_vecs[sel]
        self.labels = labels[sel]

        self.history_vecs = history_vecs  # store all; index by user_ids at __getitem__

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    def __getitem__(self, index: int):
        uid = int(self.user_ids[index])
        return {
            "query_vec": torch.from_numpy(self.query_vecs[index]),
            "history_vecs": torch.from_numpy(self.history_vecs[uid]),
            "label": torch.tensor(self.labels[index]),
        }
