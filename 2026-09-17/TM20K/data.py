from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    num_items: int = 640
    num_categories: int = 32
    seq_len: int = 96
    train_size: int = 1800
    valid_size: int = 400
    test_size: int = 400
    recent_window: int = 10
    seed: int = 0


def _build_item_categories(cfg: DataConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 11)
    item_to_cat = rng.integers(0, cfg.num_categories, size=cfg.num_items + 1, dtype=np.int64)
    item_to_cat[0] = -1
    return item_to_cat


def _sample_sequence(rng: np.random.Generator, cfg: DataConfig, item_to_cat: np.ndarray) -> np.ndarray:
    interests = rng.choice(cfg.num_categories, size=3, replace=False)
    seg_lengths = [cfg.seq_len // 3, cfg.seq_len // 3, cfg.seq_len - 2 * (cfg.seq_len // 3)]
    seq: List[int] = []
    for seg_idx, seg_len in enumerate(seg_lengths):
        active_cat = int(interests[seg_idx])
        for _ in range(seg_len):
            if rng.random() < 0.15:
                active_cat = int(rng.choice(interests))
            candidate_items = np.where(item_to_cat[1:] == active_cat)[0] + 1
            seq.append(int(rng.choice(candidate_items)))
    return np.array(seq, dtype=np.int64)


def _sample_ad_category(rng: np.random.Generator, seq: np.ndarray, cfg: DataConfig, item_to_cat: np.ndarray) -> int:
    recent_cats = item_to_cat[seq[-cfg.recent_window :]]
    long_cats = item_to_cat[seq]
    if rng.random() < 0.72:
        if rng.random() < 0.65:
            pool = recent_cats
        else:
            pool = long_cats
        return int(rng.choice(pool))
    remaining = [c for c in range(cfg.num_categories) if c not in set(long_cats.tolist())]
    if remaining:
        return int(rng.choice(remaining))
    return int(rng.integers(0, cfg.num_categories))


def _label_for(seq: np.ndarray, ad_cat: int, cfg: DataConfig, item_to_cat: np.ndarray, rng: np.random.Generator) -> int:
    cats = item_to_cat[seq]
    recent = cats[-cfg.recent_window :]
    old = cats[: -cfg.recent_window] if len(cats) > cfg.recent_window else cats

    recent_match = float(np.mean(recent == ad_cat))
    long_match = float(np.mean(cats == ad_cat))
    old_support = float(np.mean(old == ad_cat))

    recent_run = 0
    for c in recent[::-1]:
        if c == ad_cat:
            recent_run += 1
        else:
            break
    fatigue = min(1.0, recent_run / max(1, cfg.recent_window))
    novelty = float((ad_cat in old) and (ad_cat not in recent))

    score = 1.7 * recent_match + 0.85 * old_support + 0.35 * novelty - 0.30 * fatigue
    score += rng.normal(0.0, 0.10)
    threshold = 0.64 if long_match > 0.08 else 0.78
    return int(score > threshold)


class TM20KDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str) -> None:
        super().__init__()
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"unknown split={split}")
        self.cfg = cfg
        self.split = split
        self.item_to_cat = _build_item_categories(cfg)
        split_seed = {"train": 101, "valid": 202, "test": 303}[split]
        rng = np.random.default_rng(cfg.seed + split_seed)
        size = {"train": cfg.train_size, "valid": cfg.valid_size, "test": cfg.test_size}[split]

        self.samples: List[Dict[str, torch.Tensor]] = []
        for _ in range(size):
            seq = _sample_sequence(rng, cfg, self.item_to_cat)
            ad_cat = _sample_ad_category(rng, seq, cfg, self.item_to_cat)
            label = _label_for(seq, ad_cat, cfg, self.item_to_cat, rng)
            self.samples.append(
                {
                    "seq": torch.tensor(seq, dtype=torch.long),
                    "ad_cat": torch.tensor(ad_cat, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    seq = torch.stack([item["seq"] for item in batch], dim=0)
    ad_cat = torch.stack([item["ad_cat"] for item in batch], dim=0)
    label = torch.stack([item["label"] for item in batch], dim=0)
    mask = torch.ones_like(seq, dtype=torch.bool)
    return {"seq": seq, "mask": mask, "ad_cat": ad_cat, "label": label}


def binary_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.detach().cpu().flatten().numpy()
    labels = labels.detach().cpu().flatten().numpy().astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    pos_n = int(pos.sum())
    neg_n = int(neg.sum())
    if pos_n == 0 or neg_n == 0:
        return 0.5
    order = np.argsort(probs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(probs)) + 1
    pos_ranks = ranks[pos]
    auc = (pos_ranks.sum() - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)
    return float(auc)


def build_cfg_from_ckpt(payload: Dict) -> DataConfig:
    return DataConfig(**payload["data_cfg"])
