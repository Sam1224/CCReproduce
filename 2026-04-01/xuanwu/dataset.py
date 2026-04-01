from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Batch:
    image_feat: torch.Tensor
    token_ids: torch.Tensor
    label: torch.Tensor


class ToyModerationDataset(Dataset):
    """Toy multimodal moderation dataset.

    This is NOT the real Xuanwu dataset; it is a minimal dataset that matches the
    training/inference API for a multimodal content-governance classifier.
    """

    def __init__(
        self,
        n: int = 5000,
        *,
        vocab_size: int = 2048,
        seq_len: int = 32,
        image_dim: int = 512,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.image_dim = image_dim

        rng = random.Random(seed)
        self._tokens = [[rng.randrange(vocab_size) for _ in range(seq_len)] for _ in range(n)]
        self._images = [[rng.uniform(-1, 1) for _ in range(image_dim)] for _ in range(n)]

        # Label rule: a simple, learnable function of both modalities.
        self._labels = []
        for i in range(n):
            token_signal = sum(self._tokens[i]) % 3
            img_signal = 1 if (sum(self._images[i]) / image_dim) > 0 else 0
            self._labels.append(1 if (token_signal == 0 and img_signal == 1) else 0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Batch:
        return Batch(
            image_feat=torch.tensor(self._images[idx], dtype=torch.float32),
            token_ids=torch.tensor(self._tokens[idx], dtype=torch.long),
            label=torch.tensor(self._labels[idx], dtype=torch.long),
        )
