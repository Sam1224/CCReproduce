from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class KARMAToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        d_model: int = 128,
        vocab_size: int = 3000,
        min_len: int = 8,
        max_len: int = 32,
        sem_len: int = 12,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.sem_len = sem_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        l = self.rng.randint(self.min_len, self.max_len)
        behavior = torch.randn(l, self.d_model)
        text = torch.randn(l, self.d_model)
        image = torch.randn(l, self.d_model)

        # Action label (toy CTR/regression). Correlate with mean behavior.
        action_logit = behavior.mean().clamp(-2, 2)
        action_label = torch.sigmoid(action_logit) > 0.5
        action_label = action_label.float()

        sem_ids = torch.randint(0, self.vocab_size, (self.sem_len,), dtype=torch.long)
        sem_vec = torch.randn(self.d_model)

        return {
            "behavior": behavior,
            "text": text,
            "image": image,
            "action_label": action_label,
            "sem_ids": sem_ids,
            "sem_vec": sem_vec,
        }


def collate_karma(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([b["behavior"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    d_model = batch[0]["behavior"].shape[-1]

    def _pad_seq(key: str) -> torch.Tensor:
        out = torch.zeros(len(batch), max_len, d_model)
        for i, b in enumerate(batch):
            l = b[key].shape[0]
            out[i, :l] = b[key]
        return out

    behavior = _pad_seq("behavior")
    text = _pad_seq("text")
    image = _pad_seq("image")

    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths.tolist()):
        attn_mask[i, :l] = True

    sem_ids = torch.stack([b["sem_ids"] for b in batch], dim=0)
    sem_vec = torch.stack([b["sem_vec"] for b in batch], dim=0)
    action_label = torch.stack([b["action_label"] for b in batch], dim=0)

    return {
        "behavior": behavior,
        "text": text,
        "image": image,
        "attn_mask": attn_mask,
        "action_label": action_label,
        "sem_ids": sem_ids,
        "sem_vec": sem_vec,
        "lengths": lengths,
    }
