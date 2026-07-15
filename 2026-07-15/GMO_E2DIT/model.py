from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GMOConfig:
    vocab_size: int = 16
    hidden: int = 96
    num_ops: int = 5
    image_size: int = 16


class AgendaPlanner(nn.Module):
    def __init__(self, cfg: GMOConfig):
        super().__init__()
        self.text = nn.Embedding(cfg.vocab_size, cfg.hidden, padding_idx=0)
        self.image = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, cfg.hidden), nn.GELU()
        )
        self.op_head = nn.Linear(cfg.hidden * 2, cfg.num_ops)
        self.fuse = nn.Linear(cfg.hidden * 2, cfg.hidden)

    def forward(self, image: torch.Tensor, instruction_ids: torch.Tensor):
        text_feat = self.text(instruction_ids).mean(dim=1)
        image_feat = self.image(image)
        joint = torch.cat([text_feat, image_feat], dim=-1)
        return self.op_head(joint), self.fuse(joint)


class MaskGrounder(nn.Module):
    def __init__(self, cfg: GMOConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 + cfg.hidden, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 2, 1),
        )

    def forward(self, image: torch.Tensor, agenda: torch.Tensor):
        b, _, h, w = image.shape
        agenda_map = agenda[:, :, None, None].expand(b, agenda.shape[1], h, w)
        logits = self.net(torch.cat([image, agenda_map], dim=1))
        return logits[:, 0:1], logits[:, 1:2]


class MaskConditionedEditor(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(5, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 3, 1), nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor):
        return self.refine(torch.cat([image, source_mask, target_mask], dim=1))


class ReflectionHead(nn.Module):
    def __init__(self, cfg: GMOConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1)
        )

    def forward(self, image: torch.Tensor, edited: torch.Tensor):
        return self.net(torch.cat([image, edited], dim=1))


class GMOE2DITToy(nn.Module):
    def __init__(self, cfg: GMOConfig = GMOConfig()):
        super().__init__()
        self.cfg = cfg
        self.planner = AgendaPlanner(cfg)
        self.grounder = MaskGrounder(cfg)
        self.editor = MaskConditionedEditor()
        self.reflector = ReflectionHead(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]):
        op_logits, agenda = self.planner(batch["image"], batch["instruction_ids"])
        src_logits, tgt_logits = self.grounder(batch["image"], agenda)
        src_mask = torch.sigmoid(src_logits)
        tgt_mask = torch.sigmoid(tgt_logits)
        edited = self.editor(batch["image"], src_mask, tgt_mask)
        success_logits = self.reflector(batch["image"], edited)
        return {"op_logits": op_logits, "source_logits": src_logits, "target_logits": tgt_logits, "edited": edited, "success_logits": success_logits}


def loss_fn(outputs, batch):
    op_loss = F.cross_entropy(outputs["op_logits"], batch["op_id"])
    mask_weight = torch.full_like(batch["source_mask"], 12.0)
    src_loss = F.binary_cross_entropy_with_logits(outputs["source_logits"], batch["source_mask"], weight=torch.where(batch["source_mask"] > 0, mask_weight, torch.ones_like(mask_weight)))
    tgt_loss = F.binary_cross_entropy_with_logits(outputs["target_logits"], batch["target_mask"], weight=torch.where(batch["target_mask"] > 0, mask_weight, torch.ones_like(mask_weight)))
    edit_loss = F.l1_loss(outputs["edited"], batch["target"])
    refl_loss = F.binary_cross_entropy_with_logits(outputs["success_logits"], batch["success"])
    return op_loss + src_loss + tgt_loss + 2.0 * edit_loss + 0.25 * refl_loss


def mask_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return inter / max(union, 1.0)
