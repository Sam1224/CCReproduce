from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyVocab:
    pad: int = 0
    bos: int = 1
    eos: int = 2
    good: int = 3
    bad: int = 4
    a: int = 5
    b: int = 6
    c: int = 7
    vocab_size: int = 16


class ToxicNextTokenDataset(Dataset):
    def __init__(self, n: int = 4000, seq_len: int = 24, toxic_rate: float = 0.3, seed: int = 42):
        super().__init__()
        self.vocab = ToyVocab()
        self.n = n
        self.seq_len = seq_len
        self.toxic_rate = toxic_rate
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        is_toxic = self.rng.random() < self.toxic_rate
        toks = [self.vocab.bos]
        for _ in range(self.seq_len - 3):
            toks.append(self.rng.choice([self.vocab.a, self.vocab.b, self.vocab.c, self.vocab.good]))
        toks.append(self.vocab.bad if is_toxic else self.vocab.good)
        toks.append(self.vocab.eos)

        x = torch.tensor(toks[:-1], dtype=torch.long)
        y = torch.tensor(toks[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y, "is_toxic": is_toxic}


def collate(batch: list[dict]) -> dict:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    is_toxic = torch.tensor([1 if b["is_toxic"] else 0 for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "is_toxic": is_toxic}
