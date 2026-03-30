from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


Params = Dict[str, torch.Tensor]


@dataclass
class MFConfig:
    n_users: int
    n_items: int
    dim: int = 32


def init_params(cfg: MFConfig, seed: int = 0) -> Params:
    g = torch.Generator().manual_seed(seed)
    user_emb = torch.randn(cfg.n_users, cfg.dim, generator=g) * 0.01
    item_emb = torch.randn(cfg.n_items, cfg.dim, generator=g) * 0.01
    return {
        "user_emb": user_emb.requires_grad_(True),
        "item_emb": item_emb.requires_grad_(True),
    }


def predict(params: Params, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    u = F.embedding(user, params["user_emb"])
    v = F.embedding(item, params["item_emb"])
    return (u * v).sum(dim=-1)


def bce_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, label)


def batch_loss(params: Params, user: torch.Tensor, item: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    logits = predict(params, user, item)
    return bce_loss(logits, label)
