from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CemmkgOutput:
    logits: torch.Tensor
    context_gate: torch.Tensor
    alignment_scores: torch.Tensor


class ContextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0).float().unsqueeze(-1)
        pooled = (self.embedding(token_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class CEMMKGModel(nn.Module):
    def __init__(self, vocab_size: int, image_dim: int = 32, hidden_dim: int = 64, num_classes: int = 6) -> None:
        super().__init__()
        self.local_encoder = ContextEncoder(vocab_size, hidden_dim)
        self.global_encoder = ContextEncoder(vocab_size, hidden_dim)
        self.image_to_node = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.context_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, batch: Dict[str, torch.Tensor]) -> CemmkgOutput:
        visual_node = self.image_to_node(batch["feature"])
        surrounding = self.local_encoder(batch["surrounding"])
        semantic = self.local_encoder(batch["semantic"])
        global_ctx = self.global_encoder(batch["global_text"])
        local_ctx = 0.45 * surrounding + 0.55 * semantic
        gate = torch.softmax(self.context_gate(torch.cat([visual_node, local_ctx, global_ctx], dim=-1)), dim=-1)
        enriched_visual = gate[:, :1] * visual_node + gate[:, 1:2] * local_ctx + gate[:, 2:] * global_ctx
        alignment_scores = F.cosine_similarity(enriched_visual, local_ctx, dim=-1)
        logits = self.fusion(torch.cat([enriched_visual, local_ctx, global_ctx], dim=-1))
        return CemmkgOutput(logits=logits, context_gate=gate, alignment_scores=alignment_scores)


def context_alignment_loss(output: CemmkgOutput) -> torch.Tensor:
    return (1.0 - output.alignment_scores).mean().clamp_min(0.0)
