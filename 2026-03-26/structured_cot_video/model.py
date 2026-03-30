from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoTVideoModel(nn.Module):
    def __init__(self, vocab: int = 10, emb: int = 32, hidden: int = 64, num_steps: int = 5) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, batch_first=True)
        self.step_head = nn.Linear(hidden, num_steps)
        self.ans_head = nn.Linear(hidden + num_steps, 2)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        x = self.emb(video)
        _, h = self.rnn(x)
        return h.squeeze(0)

    def forward(self, video: torch.Tensor, step_onehot: torch.Tensor) -> torch.Tensor:
        h = self.encode(video)
        return self.ans_head(torch.cat([h, step_onehot], dim=-1))

    def step_logits(self, video: torch.Tensor) -> torch.Tensor:
        h = self.encode(video)
        return self.step_head(h)


def onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx, num_classes=n).float()
