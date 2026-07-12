from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EVADEConfig:
    vocab_size: int = 64
    num_rules: int = 7
    num_labels: int = 7
    text_dim: int = 96
    image_dim: int = 32
    rule_dim: int = 48
    hidden_dim: int = 128


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=4,
            dim_feedforward=text_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        embedded_text = self.embedding(text_ids)
        encoded_text = self.encoder(embedded_text)
        return encoded_text.mean(dim=1)


class ImageEncoder(nn.Module):
    def __init__(self, image_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.proj(image_features)


class RuleEncoder(nn.Module):
    def __init__(self, num_rules: int, rule_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_rules, rule_dim)

    def forward(self, rule_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(rule_id)


class EvasiveContentDetector(nn.Module):
    def __init__(self, config: EVADEConfig = EVADEConfig()):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config.vocab_size, config.text_dim)
        self.image_encoder = ImageEncoder(config.image_dim, config.hidden_dim)
        self.rule_encoder = RuleEncoder(config.num_rules, config.rule_dim)
        fusion_dim = config.text_dim + config.hidden_dim + config.rule_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.label_head = nn.Linear(config.hidden_dim, config.num_labels)
        self.evasion_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_state = self.text_encoder(batch["text_ids"])
        image_state = self.image_encoder(batch["image_features"])
        rule_state = self.rule_encoder(batch["rule_id"])
        fused_state = self.fusion(torch.cat([text_state, image_state, rule_state], dim=-1))
        return {
            "label_logits": self.label_head(fused_state),
            "evasion_logits": self.evasion_head(fused_state).squeeze(-1),
            "text_state": text_state,
            "image_state": image_state,
            "rule_state": rule_state,
        }


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    label_loss = nn.functional.cross_entropy(outputs["label_logits"], batch["label"])
    evasion_loss = nn.functional.binary_cross_entropy_with_logits(outputs["evasion_logits"], batch["is_evasive"])
    return label_loss + 0.35 * evasion_loss
