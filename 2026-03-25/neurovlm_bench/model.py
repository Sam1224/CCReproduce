from __future__ import annotations

import torch
import torch.nn as nn


class NeuroVLMProbe(nn.Module):
    def __init__(self, img_dim: int, task_vocab: int, out_dim: int = 4) -> None:
        super().__init__()
        self.task_emb = nn.Embedding(task_vocab, 16)
        self.net = nn.Sequential(
            nn.Linear(img_dim + 16, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, img: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        t = self.task_emb(task)
        return self.net(torch.cat([img, t], dim=-1))
