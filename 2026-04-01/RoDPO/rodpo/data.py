from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


@dataclass
class ToySeqData:
    num_items: int
    sequences: list[list[int]]

    @staticmethod
    def load(path: str) -> "ToySeqData":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return ToySeqData(num_items=int(obj["num_items"]), sequences=[list(map(int, s)) for s in obj["sequences"]])


class NextItemDataset(Dataset):
    def __init__(self, sequences: list[list[int]], max_seq_len: int = 30):
        self.max_seq_len = max_seq_len
        self.examples: list[tuple[list[int], int]] = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            prefix = seq[:-1]
            target = seq[-1]
            prefix = prefix[-max_seq_len:]
            self.examples.append((prefix, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def split_sequences(sequences: list[list[int]], val_ratio: float = 0.1, test_ratio: float = 0.1):
    n = len(sequences)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    train = sequences[: n - n_val - n_test]
    val = sequences[n - n_val - n_test : n - n_test]
    test = sequences[n - n_test :]
    return train, val, test


def collate_next_item(batch: Iterable[tuple[list[int], int]], pad_id: int = 0):
    batch = list(batch)
    max_len = max(len(s) for s, _ in batch)
    seq = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    tgt = torch.tensor([t for _, t in batch], dtype=torch.long)
    for i, (s, _) in enumerate(batch):
        seq[i, -len(s) :] = torch.tensor(s, dtype=torch.long)
    return seq, tgt
