from __future__ import annotations

import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, num_tasks: int, num_agents: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_tasks, num_agents))

    def forward(self, task_type: torch.Tensor) -> torch.Tensor:
        # task_type: (B,)
        return self.logits[task_type]
