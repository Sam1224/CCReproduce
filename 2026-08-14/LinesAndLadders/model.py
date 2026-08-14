from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, text_dim: int = 12, image_dim: int = 8, attr_dim: int = 6, hidden_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(text_dim + image_dim + attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([text, image, attrs], dim=-1))


class LinesAndLadders(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.encoder = ContextEncoder(hidden_dim=hidden_dim)
        self.line_agent = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.ladder_agent = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _pair_features(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        left = self.encoder(batch["left_text"], batch["left_image"], batch["left_attrs"])
        right = self.encoder(batch["right_text"], batch["right_image"], batch["right_attrs"])
        return left, right

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        left, right = self._pair_features(batch)
        features = torch.cat([left, right], dim=-1)
        return {
            "line_logit": self.line_agent(features).squeeze(-1),
            "ladder_logit": self.ladder_agent(features).squeeze(-1),
        }

    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        outputs = self.forward(batch)
        line_loss = F.binary_cross_entropy_with_logits(outputs["line_logit"], batch["label_line"])
        ladder_loss = F.binary_cross_entropy_with_logits(outputs["ladder_logit"], batch["label_ladder"])
        loss = line_loss + ladder_loss
        line_acc = ((torch.sigmoid(outputs["line_logit"]) > 0.5).float() == batch["label_line"]).float().mean().item()
        ladder_acc = ((torch.sigmoid(outputs["ladder_logit"]) > 0.5).float() == batch["label_ladder"]).float().mean().item()
        return loss, {"line_acc": line_acc, "ladder_acc": ladder_acc}

    @torch.no_grad()
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        outputs = self.forward(batch)
        result = {}
        for head, label_key in (("line_logit", "label_line"), ("ladder_logit", "label_ladder")):
            prob = torch.sigmoid(outputs[head])
            pred = (prob > 0.5).float()
            label = batch[label_key]
            tp = ((pred == 1) & (label == 1)).float().sum()
            fp = ((pred == 1) & (label == 0)).float().sum()
            fn = ((pred == 0) & (label == 1)).float().sum()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            prefix = "line" if head == "line_logit" else "ladder"
            result[f"{prefix}_precision"] = precision.item()
            result[f"{prefix}_recall"] = recall.item()
            result[f"{prefix}_f1"] = f1.item()
        return result
