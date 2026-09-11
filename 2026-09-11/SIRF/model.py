from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SirfOutput:
    cpt_logits: torch.Tensor
    verdict_logits: torch.Tensor
    rationale_logits: torch.Tensor


class SpecInternalizer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embed = nn.Embedding(4, d_model)
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

    def forward(self, token_ids: torch.Tensor, token_types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embed(token_ids) + self.type_embed(token_types) + self.position_embed(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return self.norm(hidden)


class OneModel(nn.Module):
    """SIRF-style risk foundation model.

    The implementation follows the paper's deployable shape: platform specs are
    internalized during continued pretraining, then a compact verdict-only head is
    used for online content-risk decisions.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        num_policy_nodes: int = 64,
        num_risk_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = SpecInternalizer(vocab_size, d_model, nhead, num_layers, dropout)
        self.account_proj = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.policy_head = nn.Linear(d_model, num_policy_nodes)
        self.rationale_head = nn.Linear(d_model, num_policy_nodes)
        self.verdict_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_risk_classes),
        )

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 2e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_mean(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).type_as(hidden)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> SirfOutput:
        hidden = self.backbone(batch["input_ids"], batch["token_types"], batch["attention_mask"])
        pooled = self._masked_mean(hidden, batch["attention_mask"])
        account_state = self.account_proj(batch["account_features"])
        fused = torch.cat([pooled, account_state], dim=-1)
        return SirfOutput(
            cpt_logits=self.policy_head(pooled),
            verdict_logits=self.verdict_head(fused),
            rationale_logits=self.rationale_head(pooled),
        )


def sirf_losses(out: SirfOutput, batch: Dict[str, torch.Tensor], cpt_weight: float = 0.5, rationale_weight: float = 0.25) -> Tuple[torch.Tensor, Dict[str, float]]:
    verdict = F.cross_entropy(out.verdict_logits, batch["verdict_label"])
    cpt = F.cross_entropy(out.cpt_logits, batch["policy_node"])
    rationale = F.binary_cross_entropy_with_logits(out.rationale_logits, batch["policy_path"])
    total = verdict + cpt_weight * cpt + rationale_weight * rationale
    logs = {
        "loss": float(total.detach()),
        "verdict_loss": float(verdict.detach()),
        "cpt_loss": float(cpt.detach()),
        "rationale_loss": float(rationale.detach()),
    }
    return total, logs
