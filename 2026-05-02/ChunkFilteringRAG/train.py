from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import BiEncoder, Vocab, batch_encode


@dataclass(frozen=True)
class PairExample:
    query: str
    positive: str


class PairDataset(Dataset[PairExample]):
    def __init__(self, examples: Sequence[PairExample]):
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> PairExample:
        return self._examples[index]


def collate_batch(batch: Sequence[PairExample]) -> Tuple[List[str], List[str]]:
    return [b.query for b in batch], [b.positive for b in batch]


def train_biencoder(
    model: BiEncoder,
    vocab: Vocab,
    examples: Sequence[PairExample],
    *,
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 5e-3,
    device: str = "cpu",
) -> BiEncoder:
    model.to(device)
    dataset = PairDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(epochs):
        model.train()
        for queries, positives in loader:
            q_ids = batch_encode(queries, vocab).to(device)
            p_ids = batch_encode(positives, vocab).to(device)

            q_emb = model(q_ids)  # [B, D]
            p_emb = model(p_ids)  # [B, D]

            logits = q_emb @ p_emb.T  # [B, B]
            labels = torch.arange(logits.size(0), device=device)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
