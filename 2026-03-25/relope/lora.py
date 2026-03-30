from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for a frozen Linear layer.

    y = x W^T + (alpha/r) * (x A^T) B^T

    Only A and B are trainable.
    """

    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        self.base = base
        self.cfg = cfg

        in_features = base.in_features
        out_features = base.out_features

        self.A = nn.Parameter(torch.zeros(cfg.r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, cfg.r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

        self.drop = nn.Dropout(cfg.dropout)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_d = self.drop(x)
        delta = (x_d @ self.A.t()) @ self.B.t()
        return base_out + delta * (self.cfg.alpha / self.cfg.r)
