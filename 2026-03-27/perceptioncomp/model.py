from __future__ import annotations

import torch
import torch.nn as nn


class VideoReasoner(nn.Module):
    def __init__(self, feat: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=feat, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(video)
        return self.head(h.squeeze(0))
