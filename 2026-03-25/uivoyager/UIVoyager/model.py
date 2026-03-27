from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


@dataclass
class PolicyOutput:
    logits: torch.Tensor  # (B,T,A)


class OneModel(nn.Module):
    """UI-Voyager RFT + GRSD (toy).

    Two-stage training signal:
    - RFT: learn from high-score trajectories only.
    - GRSD: for low-score trajectories, distill step logits from successful peers.
    """

    def __init__(self, d_obs: int = 128, d_model: int = 128, num_actions: int = 20, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(d_obs, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_actions)

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> PolicyOutput:
        obs = batch["obs"]  # (B,T,d_obs)
        mask = batch["mask"]  # (B,T)

        x = self.obs_proj(obs)
        x = self.encoder(x, src_key_padding_mask=~mask)
        logits = self.head(x)
        return PolicyOutput(logits=logits)


def rft_loss(out: PolicyOutput, actions: torch.Tensor, mask: torch.Tensor, traj_score: torch.Tensor, keep_threshold: float = 0.5) -> torch.Tensor:
    keep = (traj_score >= keep_threshold).unsqueeze(1) & mask  # (B,T)
    if keep.sum() == 0:
        return torch.tensor(0.0, device=actions.device)
    return F.cross_entropy(out.logits[keep], actions[keep])


def grsd_loss(
    out: PolicyOutput,
    peer_logits: torch.Tensor,
    mask: torch.Tensor,
    traj_score: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    # Distill only for low-score trajectories.
    low = (traj_score < 0.5).unsqueeze(1) & mask
    if low.sum() == 0:
        return torch.tensor(0.0, device=peer_logits.device)

    p = F.log_softmax(out.logits / temperature, dim=-1)
    q = F.softmax(peer_logits / temperature, dim=-1)
    kl = F.kl_div(p, q, reduction="none").sum(dim=-1) * (temperature**2)
    return kl[low].mean()
