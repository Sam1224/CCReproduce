from __future__ import annotations

import torch
import torch.nn as nn


class MicroActionModel(nn.Module):
    def __init__(self, vocab: int = 20, emb: int = 48, hidden: int = 96, L: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, vocab * 2)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.emb(seq)
        out, _ = self.rnn(x)
        center = out.shape[1] // 2
        return self.head(out[:, center])
