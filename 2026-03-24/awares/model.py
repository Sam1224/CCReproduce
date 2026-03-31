from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AwaResConfig:
    num_actions: int = 10  # 0=no_crop, 1..9=crop index + 1
    num_classes: int = 10
    hidden: int = 64


class ConvEncoder(nn.Module):
    def __init__(self, *, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AwaResModel(nn.Module):
    def __init__(self, cfg: AwaResConfig):
        super().__init__()
        self.cfg = cfg
        self.low_enc = ConvEncoder(hidden=cfg.hidden)
        self.crop_enc = ConvEncoder(hidden=cfg.hidden)

        self.action_head = nn.Linear(cfg.hidden, cfg.num_actions)
        self.answer_low_head = nn.Linear(cfg.hidden, cfg.num_classes)
        self.answer_fused_head = nn.Linear(cfg.hidden * 2, cfg.num_classes)

    def forward_low(self, img_low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.low_enc(img_low)
        action_logits = self.action_head(feat)
        ans_logits = self.answer_low_head(feat)
        return action_logits, ans_logits, feat

    def forward_with_crop(self, low_feat: torch.Tensor, img_crop: torch.Tensor) -> torch.Tensor:
        crop_feat = self.crop_enc(img_crop)
        fused = torch.cat([low_feat, crop_feat], dim=-1)
        return self.answer_fused_head(fused)
