from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvadeConfig:
    vocab_size: int = 18
    num_classes: int = 6
    hidden: int = 96


class TextEvidenceEncoder(nn.Module):
    def __init__(self, config: EvadeConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden, padding_idx=0)
        self.gate = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.GELU(), nn.Linear(config.hidden, config.hidden), nn.Sigmoid())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = (token_ids != 0).float().unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled * self.gate(pooled)


class VisualEvidenceEncoder(nn.Module):
    def __init__(self, config: EvadeConfig):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, config.hidden, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, image: torch.Tensor):
        feature_map = self.features(image)
        pooled = self.pool(feature_map).flatten(1)
        return pooled, feature_map


class CrossModalConsistencyHead(nn.Module):
    def __init__(self, config: EvadeConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(config.hidden * 4, config.hidden), nn.GELU(), nn.Linear(config.hidden, 1))

    def forward(self, text_feature: torch.Tensor, image_feature: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([text_feature, image_feature, torch.abs(text_feature - image_feature), text_feature * image_feature], dim=-1)
        return self.net(joint)


class EvadeBenchToyModel(nn.Module):
    def __init__(self, config: EvadeConfig = EvadeConfig()):
        super().__init__()
        self.config = config
        self.text_encoder = TextEvidenceEncoder(config)
        self.visual_encoder = VisualEvidenceEncoder(config)
        self.fusion = nn.Sequential(nn.Linear(config.hidden * 2, config.hidden), nn.GELU(), nn.Dropout(0.1))
        self.category_head = nn.Linear(config.hidden, config.num_classes)
        self.violation_head = nn.Linear(config.hidden, 1)
        self.consistency_head = CrossModalConsistencyHead(config)
        self.evasion_decoder = nn.Sequential(
            nn.Conv2d(config.hidden, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        text_feature = self.text_encoder(batch["token_ids"])
        image_feature, feature_map = self.visual_encoder(batch["image"])
        fused = self.fusion(torch.cat([text_feature, image_feature], dim=-1))
        return {
            "category_logits": self.category_head(fused),
            "violation_logits": self.violation_head(fused),
            "consistency_logits": self.consistency_head(text_feature, image_feature),
            "evasion_logits": self.evasion_decoder(feature_map),
        }


def loss_fn(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    category_loss = F.cross_entropy(outputs["category_logits"], batch["label"])
    violation_loss = F.binary_cross_entropy_with_logits(outputs["violation_logits"], batch["violation"])
    consistency_loss = F.binary_cross_entropy_with_logits(outputs["consistency_logits"], batch["consistency"])
    mask_weight = torch.where(batch["evasion_mask"] > 0, torch.full_like(batch["evasion_mask"], 10.0), torch.ones_like(batch["evasion_mask"]))
    evasion_loss = F.binary_cross_entropy_with_logits(outputs["evasion_logits"], batch["evasion_mask"], weight=mask_weight)
    return category_loss + violation_loss + 0.5 * consistency_loss + 0.7 * evasion_loss


def binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    return (prediction == target).float().mean().item()


def mask_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = (torch.sigmoid(logits) > 0.5).float()
    intersection = (prediction * target).sum().item()
    union = ((prediction + target) > 0).float().sum().item()
    return intersection / max(union, 1.0)
