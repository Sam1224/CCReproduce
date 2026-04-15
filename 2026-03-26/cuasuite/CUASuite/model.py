from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


@dataclass
class CUAPolicyOutput:
    action_logits: torch.Tensor  # (B,A)


class FrameEncoder(nn.Module):
    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        h = self.net(x).flatten(1)
        return h


class OneModel(nn.Module):
    """Toy computer-use agent for CUA-Suite."""

    def __init__(self, d_model: int = 128, num_actions: int = 12) -> None:
        super().__init__()
        self.frame_encoder = FrameEncoder(d_model)
        self.temporal = nn.GRU(d_model, d_model, batch_first=True)
        self.cursor_mlp = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, num_actions))

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> CUAPolicyOutput:
        frames = batch["frames"]  # (B,T,C,H,W)
        cursor_xy = batch["cursor_xy"]  # (B,2)

        bsz, t = frames.shape[0], frames.shape[1]
        frames_flat = frames.reshape(bsz * t, 3, frames.shape[-2], frames.shape[-1])
        f = self.frame_encoder(frames_flat).reshape(bsz, t, -1)
        _, last = self.temporal(f)
        video_vec = last.squeeze(0)

        cursor_vec = self.cursor_mlp(cursor_xy)
        action_logits = self.head(torch.cat([video_vec, cursor_vec], dim=-1))
        return CUAPolicyOutput(action_logits=action_logits)


def imitation_loss(out: CUAPolicyOutput, action_id: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = F.cross_entropy(out.action_logits, action_id)
    return loss, {"ce": float(loss.detach())}
