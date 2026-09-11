from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PtcOutput:
    logits: torch.Tensor
    task_attention: torch.Tensor


class TaskConditionedEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_tasks: int, d_model: int, nhead: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.task_embed = nn.Embedding(num_tasks, d_model)
        self.position_embed = nn.Embedding(512, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, item_ids: torch.Tensor, task_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = item_ids.shape
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, -1)
        task = self.task_embed(task_id).unsqueeze(1)
        hidden = self.token_embed(item_ids) + self.position_embed(positions) + task
        return self.norm(self.encoder(hidden, src_key_padding_mask=~mask))


class OneModel(nn.Module):
    """Positional Task Conditioning for product-family defect detection.

    Each task is encoded as an explicit task embedding added at every product
    position, matching the paper's repeated task-conditioning idea for one
    shared student model across defect subtasks.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        num_tasks: int = 4,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TaskConditionedEncoder(vocab_size, num_tasks, d_model, nhead, num_layers, dropout)
        self.item_score = nn.Linear(d_model, 1)
        self.family_head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3))

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 2e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, batch: Dict[str, torch.Tensor]) -> PtcOutput:
        hidden = self.encoder(batch["item_ids"], batch["task_id"], batch["item_mask"])
        raw_scores = self.item_score(hidden).squeeze(-1).masked_fill(~batch["item_mask"], -1e4)
        attention = torch.softmax(raw_scores, dim=-1)
        weighted = torch.bmm(attention.unsqueeze(1), hidden).squeeze(1)
        masked = hidden.masked_fill(~batch["item_mask"].unsqueeze(-1), 0.0)
        mean = masked.sum(dim=1) / batch["item_mask"].sum(dim=1, keepdim=True).clamp_min(1)
        max_pool = hidden.masked_fill(~batch["item_mask"].unsqueeze(-1), -1e4).max(dim=1).values
        logits = self.family_head(torch.cat([weighted, mean, max_pool], dim=-1))
        return PtcOutput(logits=logits, task_attention=attention)


def ptc_losses(out: PtcOutput, batch: Dict[str, torch.Tensor], teacher_weight: float = 0.35) -> Tuple[torch.Tensor, Dict[str, float]]:
    supervised = F.cross_entropy(out.logits, batch["label"])
    teacher = F.kl_div(
        F.log_softmax(out.logits, dim=-1),
        batch["teacher_probs"],
        reduction="batchmean",
    )
    total = supervised + teacher_weight * teacher
    return total, {"loss": float(total.detach()), "ce": float(supervised.detach()), "teacher_kl": float(teacher.detach())}
