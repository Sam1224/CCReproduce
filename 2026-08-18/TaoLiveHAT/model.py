from dataclasses import dataclass

import torch
from torch import nn


class HarnessAwarePolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, num_labels: int = 4):
        super().__init__()
        self.semantic_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.tool_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.hook_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 16 + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic_encoder(x[:, :8])
        prompt = self.prompt_encoder(x[:, 8:12])
        tool = self.tool_encoder(x[:, 12:16])
        hook = self.hook_encoder(x[:, 16:20])
        route = self.route_encoder(x[:, 20:24])
        fused = torch.cat([semantic, prompt, tool, hook, route], dim=-1)
        return self.fusion(fused)


@dataclass
class ForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class TaoLiveHATModel(nn.Module):
    def __init__(self, input_dim: int = 24, num_labels: int = 4):
        super().__init__()
        self.policy = HarnessAwarePolicy(input_dim=input_dim, num_labels=num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> ForwardOutput:
        logits = self.policy(x)
        if y is None:
            loss = torch.zeros((), device=logits.device)
        else:
            loss = self.loss_fn(logits, y)
        return ForwardOutput(loss=loss, logits=logits)
