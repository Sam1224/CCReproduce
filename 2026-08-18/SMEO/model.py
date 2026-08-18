from dataclasses import dataclass

import torch
from torch import nn


class TrajectoryUtilityModel(nn.Module):
    def __init__(self, input_dim: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurvivalAwareRanker(nn.Module):
    def __init__(self, input_dim: int = 89, num_assets: int = 6):
        super().__init__()
        self.num_assets = num_assets
        self.asset_proj = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(17, 24),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x[:, :17]
        asset_blob = x[:, 17:]
        assets = asset_blob.view(x.size(0), self.num_assets, 12)
        state_repr = self.state_proj(state).unsqueeze(1).expand(-1, self.num_assets, -1)
        asset_repr = self.asset_proj(assets)
        logits = self.scorer(torch.cat([state_repr, asset_repr], dim=-1)).squeeze(-1)
        return logits


@dataclass
class LossOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class SMEOPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.utility = TrajectoryUtilityModel()
        self.ranker = SurvivalAwareRanker()
        self.utility_loss = nn.MSELoss()
        self.rank_loss = nn.CrossEntropyLoss()

    def forward_utility(self, x: torch.Tensor, y: torch.Tensor | None = None) -> LossOutput:
        pred = self.utility(x)
        loss = torch.zeros((), device=pred.device) if y is None else self.utility_loss(pred, y)
        return LossOutput(loss=loss, logits=pred)

    def forward_rank(self, x: torch.Tensor, y: torch.Tensor | None = None) -> LossOutput:
        logits = self.ranker(x)
        loss = torch.zeros((), device=logits.device) if y is None else self.rank_loss(logits, y)
        return LossOutput(loss=loss, logits=logits)
