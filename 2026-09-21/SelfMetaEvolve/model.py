from __future__ import annotations

import torch
from torch import nn


class SelfMetaEvolve(nn.Module):
    def __init__(self, num_roles: int, num_fields: int, dim: int = 32):
        super().__init__()
        self.role_embedding = nn.Embedding(num_roles, dim)
        self.meta_prompt = nn.Parameter(torch.zeros(dim))
        self.editor = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh(), nn.Linear(dim, num_fields))

    def forward(self, role_id: torch.Tensor, feedback_memory: torch.Tensor | None = None) -> torch.Tensor:
        role_state = self.role_embedding(role_id)
        if feedback_memory is None:
            feedback_memory = torch.zeros_like(role_state)
        meta = self.meta_prompt.unsqueeze(0).expand_as(role_state)
        return self.editor(torch.cat([role_state + feedback_memory, meta], dim=-1))

    def evolve_meta_prompt(self, successful_edits: torch.Tensor, momentum: float = 0.15) -> None:
        if successful_edits.numel() == 0:
            return
        update = successful_edits.mean(dim=0)
        with torch.no_grad():
            self.meta_prompt.mul_(1.0 - momentum).add_(momentum * update)
