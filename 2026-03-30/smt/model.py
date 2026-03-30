from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from blocks import (
    CatEmbedding,
    ConcatInteraction,
    DotInteraction,
    MLP,
    TemperatureCalibrator,
)


Interaction = Literal["dot", "concat"]


@dataclass
class SMTConfig:
    n_cat: int = 50
    cat_dim: int = 16
    x_dim: int = 8
    x_proj: int = 16
    interaction: Interaction = "concat"
    head_hidden: int = 64
    use_calibration: bool = True


class StandardModelTemplate(nn.Module):
    """A tiny Standard Model Template.

    It exposes stable block interfaces so many models can share the same code path while
    swapping in different blocks/configs.
    """

    def __init__(self, cfg: SMTConfig):
        super().__init__()
        self.cfg = cfg

        self.cat = CatEmbedding(cfg.n_cat, cfg.cat_dim)
        self.x_proj = nn.Linear(cfg.x_dim, cfg.x_proj)

        if cfg.interaction == "dot":
            self.interaction = DotInteraction()
            head_in = 1
        else:
            self.interaction = ConcatInteraction()
            head_in = cfg.cat_dim + cfg.x_proj

        self.head = MLP(head_in, hidden=cfg.head_hidden, out_dim=1, dropout=0.1)
        self.calib = TemperatureCalibrator(1.0) if cfg.use_calibration else None

    def forward(self, cat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = self.cat(cat)
        b = self.x_proj(x)
        z = self.interaction(a, b)
        logits = self.head(z).squeeze(-1)
        if self.calib is not None:
            logits = self.calib(logits)
        return logits
