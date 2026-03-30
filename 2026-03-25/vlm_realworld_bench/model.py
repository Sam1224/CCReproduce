from __future__ import annotations

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, img_dim: int, q_vocab: int, num_answers: int) -> None:
        super().__init__()
        self.q_emb = nn.Embedding(q_vocab, 16)
        self.net = nn.Sequential(
            nn.Linear(img_dim + 16, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, num_answers),
        )

    def forward(self, img: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        qe = self.q_emb(q)
        x = torch.cat([img, qe], dim=-1)
        return self.net(x)
