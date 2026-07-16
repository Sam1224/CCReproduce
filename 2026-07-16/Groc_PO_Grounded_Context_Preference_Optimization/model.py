from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GrocPOConfig:
    vocab_size: int = 64
    image_dim: int = 8
    hidden: int = 96
    num_stages: int = 3
    beta: float = 0.2


class GroundedContextEncoder(nn.Module):
    def __init__(self, cfg: GrocPOConfig):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.hidden, padding_idx=0)
        self.stage = nn.Embedding(cfg.num_stages, cfg.hidden)
        self.image = nn.Sequential(nn.Linear(cfg.image_dim, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.context_gate = nn.Sequential(nn.Linear(cfg.hidden * 3, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.hidden), nn.Sigmoid())
        self.reward = nn.Sequential(nn.Linear(cfg.hidden * 3, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).float().unsqueeze(-1)
        emb = self.token(ids) * mask
        return emb.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def score(self, image: torch.Tensor, prompt: torch.Tensor, stage: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        image_feat = self.image(image)
        prompt_feat = self.encode_text(prompt)
        response_feat = self.encode_text(response)
        stage_feat = self.stage(stage)
        grounded_context = torch.cat([image_feat + stage_feat, prompt_feat, response_feat], dim=-1)
        gate = self.context_gate(grounded_context)
        fused_grounding = gate * image_feat + (1.0 - gate) * prompt_feat
        return self.reward(torch.cat([fused_grounding, response_feat, stage_feat], dim=-1)).squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        chosen_score = self.score(batch["image"], batch["prompt"], batch["stage"], batch["chosen"])
        rejected_score = self.score(batch["image"], batch["prompt"], batch["stage"], batch["rejected"])
        return {"chosen_score": chosen_score, "rejected_score": rejected_score, "margin": chosen_score - rejected_score}


def groc_po_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], beta: float = 0.2) -> torch.Tensor:
    signed_margin = torch.where(batch["label"] > 0.5, outputs["margin"], -outputs["margin"])
    preference_loss = -F.logsigmoid(beta * signed_margin).mean()
    calibration = F.binary_cross_entropy_with_logits(outputs["margin"], batch["label"])
    return preference_loss + 0.1 * calibration


def preference_accuracy(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
    pred = (outputs["margin"] > 0).float()
    return (pred == batch["label"]).float().mean().item()
