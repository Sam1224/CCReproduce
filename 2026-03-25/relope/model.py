from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import LoRAConfig, LoRALinear


@dataclass
class ModelConfig:
    d_in: int = 32
    d_hidden: int = 64


class BaseMultimodal(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.d_in, cfg.d_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.GELU(),
        )
        self.classifier = nn.Linear(cfg.d_hidden, 2)

    def forward(self, x_text: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_text, x_img], dim=-1)
        h = self.encoder(x)
        return self.classifier(h)


class LoRAProbe(nn.Module):
    """A probe that reuses the frozen base encoder and adds LoRA on the classifier."""

    def __init__(self, base: BaseMultimodal, lora_cfg: LoRAConfig):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.lora_head = LoRALinear(self.base.classifier, lora_cfg)

    def forward(self, x_text: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_text, x_img], dim=-1)
        h = self.base.encoder(x)
        return self.lora_head(h)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def kl_probe_to_base(probe_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """KL(probe || base) on per-sample distributions."""
    p = F.log_softmax(probe_logits, dim=-1)
    q = F.softmax(base_logits.detach(), dim=-1)
    return F.kl_div(p, q, reduction='batchmean')


def route_by_nll_improvement(
    base_logits: torch.Tensor,
    probe_logits: Dict[int, torch.Tensor],
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the probe with best NLL improvement (lower is better).

    Returns:
      - chosen logits
      - chosen domain id (=-1 means base)
    """
    base_nll = F.cross_entropy(base_logits, y, reduction='none')
    best_nll = base_nll
    best_logits = base_logits
    best_dom = torch.full_like(y, fill_value=-1)

    for dom, lg in probe_logits.items():
        nll = F.cross_entropy(lg, y, reduction='none')
        improved = nll < best_nll
        best_nll = torch.where(improved, nll, best_nll)
        best_dom = torch.where(improved, torch.full_like(best_dom, dom), best_dom)
        best_logits = torch.where(improved.unsqueeze(-1), lg, best_logits)

    return best_logits, best_dom
