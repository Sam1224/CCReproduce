from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class VerifierToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        vocab_size: int = 5000,
        min_len: int = 128,
        max_len: int = 256,
        seed: int = 123,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        length = self.rng.randint(self.min_len, self.max_len)
        input_ids = torch.randint(0, self.vocab_size, (length,), dtype=torch.long)

        halluc = torch.tensor(self.rng.random() < 0.5, dtype=torch.float32)

        # Unsupported span mask (token-level), correlated with halluc label.
        span_labels = torch.zeros(length, dtype=torch.float32)
        if halluc.item() == 1.0:
            # Add 1-3 random spans.
            for _ in range(self.rng.randint(1, 3)):
                start = self.rng.randint(0, max(0, length - 16))
                end = min(length, start + self.rng.randint(4, 24))
                span_labels[start:end] = 1.0

        return {"input_ids": input_ids, "span_labels": span_labels, "halluc": halluc}


def collate_verifier(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([b["input_ids"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    input_ids = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    span_labels = torch.zeros((len(batch), max_len), dtype=torch.float32)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    halluc = torch.stack([b["halluc"] for b in batch], dim=0)

    for i, sample in enumerate(batch):
        l = sample["input_ids"].shape[0]
        input_ids[i, :l] = sample["input_ids"]
        span_labels[i, :l] = sample["span_labels"]
        attn_mask[i, :l] = True

    return {
        "input_ids": input_ids,
        "span_labels": span_labels,
        "halluc": halluc,
        "attn_mask": attn_mask,
        "lengths": lengths,
    }
