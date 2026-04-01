from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ReLogOutput:
    level_logits: torch.Tensor
    template_logits: torch.Tensor


class ReLogClassifier(nn.Module):
    """A lightweight proxy model for ReLog-style logging generation.

    The original paper uses an LLM to generate and refine logging statements using
    runtime feedback. This reproduction provides a small PyTorch baseline that
    learns to pick (1) a log level and (2) a template ID from a toy dataset.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_levels: int,
        num_templates: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.level_head = nn.Linear(hidden_size, num_levels)
        self.template_head = nn.Linear(hidden_size, num_templates)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> ReLogOutput:
        embedded = self.embedding(token_ids)
        embedded = embedded * attention_mask.unsqueeze(-1)
        encoded, _ = self.encoder(embedded)
        pooled = (encoded * attention_mask.unsqueeze(-1)).sum(dim=1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = pooled / denom
        return ReLogOutput(level_logits=self.level_head(pooled), template_logits=self.template_head(pooled))
