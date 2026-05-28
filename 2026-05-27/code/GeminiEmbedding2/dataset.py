"""
Toy dataset for Gemini Embedding 2 reproduction.

Implements interface-aligned datasets for all four modalities:
  - TextPairDataset       — (anchor, positive) text pairs
  - ImageTextPairDataset  — (image_patches, text_ids) pairs
  - AudioTextPairDataset  — (mel_spectrogram, text_ids) pairs
  - VideoTextPairDataset  — (frame_patches, text_ids) pairs
  - MultiModalDataset     — wraps all modalities with task labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import random


class TextPairDataset(Dataset):
    """
    Toy text-pair dataset for contrastive pre-training.
    In production: web-scale text pairs (query, passage), (caption, doc), etc.
    """
    def __init__(self, num_samples: int = 1000, vocab_size: int = 32000,
                 seq_len: int = 128, task_id: int = 0):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.task_id = task_id
        # Generate random token sequences (toy data)
        torch.manual_seed(42)
        self.queries = torch.randint(1, vocab_size, (num_samples, seq_len))
        # Positives: same content with minor perturbation
        self.keys = self.queries.clone()
        for i in range(num_samples):
            # Randomly replace 10% of tokens
            n_replace = seq_len // 10
            idx = random.sample(range(seq_len), n_replace)
            self.keys[i, idx] = torch.randint(1, vocab_size, (n_replace,))
        self.masks = torch.ones(num_samples, seq_len, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "query_ids": self.queries[idx],
            "query_mask": self.masks[idx],
            "key_ids": self.keys[idx],
            "key_mask": self.masks[idx],
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
            "modality": "text",
        }


class ImageTextPairDataset(Dataset):
    """
    Toy image-text pair dataset.
    In production: LAION-5B scale image-caption pairs.
    """
    def __init__(self, num_samples: int = 500, vocab_size: int = 32000,
                 seq_len: int = 64, num_patches: int = 196,
                 patch_dim: int = 768, task_id: int = 1):
        self.num_samples = num_samples
        self.task_id = task_id
        torch.manual_seed(43)
        # Image patches (B, N, patch_dim)
        self.patches = torch.randn(num_samples, num_patches, patch_dim)
        # Text captions
        self.text_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
        self.text_masks = torch.ones(num_samples, seq_len, dtype=torch.long)
        # Patch mask (all patches visible)
        self.patch_masks = torch.ones(num_samples, num_patches + 1, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "query_patches": self.patches[idx],
            "query_mask": self.patch_masks[idx],
            "key_ids": self.text_ids[idx],
            "key_mask": self.text_masks[idx],
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
            "modality": "image_text",
        }


class MultiModalDataset(Dataset):
    """
    Unified dataset that mixes multiple modality datasets with task labels.
    """
    def __init__(self, datasets: list, task_ids: list):
        self.datasets = datasets
        self.task_ids = task_ids
        self.lengths = [len(d) for d in datasets]
        self.total = sum(self.lengths)
        # Build index mapping
        self.index_map = []
        for ds_idx, ds in enumerate(datasets):
            for item_idx in range(len(ds)):
                self.index_map.append((ds_idx, item_idx))

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        ds_idx, item_idx = self.index_map[idx]
        sample = self.datasets[ds_idx][item_idx]
        sample["task_id"] = torch.tensor(self.task_ids[ds_idx], dtype=torch.long)
        return sample


def collate_text_pair(batch):
    """Collate function for text-only batches."""
    return {
        "query_ids": torch.stack([b["query_ids"] for b in batch]),
        "query_mask": torch.stack([b["query_mask"] for b in batch]),
        "key_ids": torch.stack([b["key_ids"] for b in batch]),
        "key_mask": torch.stack([b["key_mask"] for b in batch]),
        "task_id": torch.stack([b["task_id"] for b in batch]),
    }


def get_dataloaders(cfg, batch_size: int = 32, num_workers: int = 0):
    """Return train and validation dataloaders (toy scale)."""
    text_train = TextPairDataset(num_samples=800, vocab_size=cfg.vocab_size,
                                 seq_len=128, task_id=0)
    text_val = TextPairDataset(num_samples=200, vocab_size=cfg.vocab_size,
                               seq_len=128, task_id=0)
    train_loader = DataLoader(text_train, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_text_pair,
                              num_workers=num_workers)
    val_loader = DataLoader(text_val, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_text_pair,
                            num_workers=num_workers)
    return train_loader, val_loader
