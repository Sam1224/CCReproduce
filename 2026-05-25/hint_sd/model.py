from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PolicyOutput:
    logits: torch.Tensor  # [B, 4]


class PolicyNet(nn.Module):
    """A lightweight policy network.

    Student: takes state only.
    Teacher (privileged): takes state + feedback_action embedding.

    This mirrors the paper's idea: teacher is conditioned on hindsight feedback,
    while student must internalize the behavior without feedback.
    """

    def __init__(self, state_dim: int = 5, hidden_dim: int = 128, use_feedback: bool = False):
        super().__init__()
        self.use_feedback = use_feedback

        feedback_dim = 8 if use_feedback else 0
        if use_feedback:
            self.feedback_emb = nn.Embedding(4, feedback_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + feedback_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, states: torch.Tensor, feedback_action: Optional[torch.Tensor] = None) -> PolicyOutput:
        if self.use_feedback:
            if feedback_action is None:
                raise ValueError("feedback_action is required when use_feedback=True")
            fb = self.feedback_emb(feedback_action)
            x = torch.cat([states, fb], dim=-1)
        else:
            x = states
        return PolicyOutput(logits=self.net(x))
