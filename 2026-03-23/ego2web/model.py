from __future__ import annotations

import torch
import torch.nn as nn


class EgoEncoder(nn.Module):
    def __init__(self, vocab: int, emb: int = 64, hidden: int = 96, num_tasks: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(input_size=emb, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.emb(seq)
        _, h = self.rnn(x)
        return self.head(h.squeeze(0))
