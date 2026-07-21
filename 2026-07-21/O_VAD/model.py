from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OvadConfig:
    feat_dim: int = 8
    hidden: int = 64
    num_states: int = 5
    num_types: int = 4


class ObjectTracker(nn.Module):
    def __init__(self, cfg: OvadConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.feat_dim, cfg.hidden)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=2, batch_first=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, objects, dim = frames.shape
        flat = frames.permute(0, 2, 1, 3).reshape(batch * objects, steps, dim)
        tracked, _ = self.gru(self.in_proj(flat))
        return tracked.reshape(batch, objects, steps, -1).permute(0, 2, 1, 3)


class StateReasoner(nn.Module):
    def __init__(self, cfg: OvadConfig):
        super().__init__()
        self.state_head = nn.Linear(cfg.hidden, cfg.num_states)
        self.frame_head = nn.Linear(cfg.hidden, 1)
        self.object_gate = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))
        self.video_head = nn.Linear(cfg.hidden, 1)
        self.type_head = nn.Linear(cfg.hidden, cfg.num_types)
        self.object_head = nn.Linear(cfg.hidden, 1)

    def forward(self, tracks: torch.Tensor) -> Dict[str, torch.Tensor]:
        state_logits = self.state_head(tracks)
        frame_repr = tracks.mean(dim=2)
        frame_logits = self.frame_head(frame_repr)
        object_scores = self.object_gate(tracks).squeeze(-1)
        object_attn = torch.softmax(object_scores, dim=2)
        object_summary = (tracks * object_attn.unsqueeze(-1)).sum(dim=2)
        video_context = object_summary.max(dim=1).values
        pooled_objects = tracks.mean(dim=1)
        return {
            "state_logits": state_logits,
            "frame_logits": frame_logits,
            "video_logits": self.video_head(video_context),
            "type_logits": self.type_head(video_context),
            "object_logits": self.object_head(pooled_objects).squeeze(-1),
        }


class OvadToy(nn.Module):
    def __init__(self, cfg: OvadConfig = OvadConfig()):
        super().__init__()
        self.cfg = cfg
        self.tracker = ObjectTracker(cfg)
        self.reasoner = StateReasoner(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tracks = self.tracker(batch["frames"])
        outputs = self.reasoner(tracks)
        outputs["tracks"] = tracks
        return outputs


def frame_iou(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).sum().item()
    union = ((pred + targets) > 0).float().sum().item()
    return inter / max(union, 1.0)


def loss_fn(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    video_loss = F.binary_cross_entropy_with_logits(outputs["video_logits"], batch["video_label"])
    frame_loss = F.binary_cross_entropy_with_logits(outputs["frame_logits"], batch["frame_mask"])
    state_loss = F.cross_entropy(outputs["state_logits"].reshape(-1, outputs["state_logits"].shape[-1]), batch["states"].reshape(-1))
    type_loss = F.cross_entropy(outputs["type_logits"], batch["anomaly_type"])

    object_targets = batch["anomaly_object"].clone()
    negative_mask = object_targets < 0
    object_targets[negative_mask] = 0
    object_loss = F.cross_entropy(outputs["object_logits"], object_targets)
    object_loss = object_loss * (1.0 - negative_mask.float().mean() * 0.5)
    return video_loss + frame_loss + 0.5 * state_loss + 0.5 * type_loss + 0.25 * object_loss


def build_report(outputs: Dict[str, torch.Tensor], index: int = 0) -> Dict[str, object]:
    frame_probs = torch.sigmoid(outputs["frame_logits"][index]).squeeze(-1)
    anomalous = bool(torch.sigmoid(outputs["video_logits"][index]).item() > 0.5)
    object_id = int(outputs["object_logits"][index].argmax().item())
    anomaly_type = int(outputs["type_logits"][index].argmax().item())
    active = torch.where(frame_probs > 0.5)[0]
    if active.numel() == 0:
        segment = []
    else:
        segment = [int(active.min().item()), int(active.max().item())]
    return {
        "overall_anomaly": anomalous,
        "anomaly_object": object_id,
        "anomaly_type_id": anomaly_type,
        "anomaly_segment": segment,
        "reasoning": "Track object-wise state transitions, identify abrupt state jumps, and summarize the anomalous object with temporal evidence.",
    }
