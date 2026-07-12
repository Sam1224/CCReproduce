from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model import EvasiveContentDetector


class VisualDescriber(nn.Module):
    def __init__(self, detector: EvasiveContentDetector):
        super().__init__()
        self.image_encoder = detector.image_encoder
        self.visual_head = nn.Linear(detector.config.hidden_dim, detector.config.num_labels)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        visual_state = self.image_encoder(image_features)
        return self.visual_head(visual_state)


class RuleReasoner(nn.Module):
    def __init__(self, detector: EvasiveContentDetector):
        super().__init__()
        self.rule_encoder = detector.rule_encoder
        self.rule_head = nn.Linear(detector.config.rule_dim, detector.config.num_labels)

    def forward(self, rule_id: torch.Tensor) -> torch.Tensor:
        rule_state = self.rule_encoder(rule_id)
        return self.rule_head(rule_state)


class DetectionCoordinator(nn.Module):
    def __init__(self, detector: EvasiveContentDetector):
        super().__init__()
        self.detector = detector
        self.visual_agent = VisualDescriber(detector)
        self.rule_agent = RuleReasoner(detector)
        self.coordinator = nn.Linear(detector.config.num_labels * 3, detector.config.num_labels)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        detector_outputs = self.detector(batch)
        visual_logits = self.visual_agent(batch["image_features"])
        rule_logits = self.rule_agent(batch["rule_id"])
        joint_logits = self.coordinator(torch.cat([detector_outputs["label_logits"], visual_logits, rule_logits], dim=-1))
        return {
            "label_logits": joint_logits,
            "visual_logits": visual_logits,
            "rule_logits": rule_logits,
            "evasion_logits": detector_outputs["evasion_logits"],
        }


def multi_agent_predict(detector: EvasiveContentDetector, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    coordinator = DetectionCoordinator(detector).to(next(detector.parameters()).device)
    coordinator.eval()
    with torch.no_grad():
        outputs = coordinator(batch)
        return outputs["label_logits"].argmax(dim=-1)
