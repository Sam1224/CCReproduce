from __future__ import annotations

import torch
from torch import nn

from data import ad_to_vector, dialog_to_bow, relevance_label, slate_reward, tool_features


class OpportunityGateNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class RelevanceJudge(nn.Module):
    def __init__(self, dialog_dim: int, ad_dim: int, tool_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dialog_dim + ad_dim + tool_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dialog_features: torch.Tensor, ad_features: torch.Tensor, tool_features_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([dialog_features, ad_features, tool_features_tensor], dim=-1)).squeeze(-1)


class SlateOrchestrator(nn.Module):
    def __init__(self, dialog_dim: int, ad_dim: int, tool_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dialog_encoder = nn.Sequential(
            nn.Linear(dialog_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.ad_encoder = nn.Sequential(
            nn.Linear(ad_dim + tool_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dialog_features: torch.Tensor, ad_features: torch.Tensor, tool_features_tensor: torch.Tensor) -> torch.Tensor:
        dialog_hidden = self.dialog_encoder(dialog_features)
        ad_hidden = self.ad_encoder(torch.cat([ad_features, tool_features_tensor], dim=-1))
        return self.scorer(torch.cat([dialog_hidden, ad_hidden], dim=-1)).squeeze(-1)


class AdsWorldEnginePipeline:
    def __init__(self, gate: OpportunityGateNet, judge: RelevanceJudge, orchestrator: SlateOrchestrator, ads: list) -> None:
        self.gate = gate
        self.judge = judge
        self.orchestrator = orchestrator
        self.ads = ads

    @torch.no_grad()
    def predict(self, example, top_k: int = 3):
        dialog_features = dialog_to_bow(example).unsqueeze(0)
        gate_prob = torch.sigmoid(self.gate(dialog_features)).item()
        if gate_prob < 0.5:
            return gate_prob, []
        scored = []
        for ad in self.ads:
            ad_features = ad_to_vector(ad).unsqueeze(0)
            tool_tensor = tool_features(example, ad).unsqueeze(0)
            judge_score = torch.sigmoid(self.judge(dialog_features, ad_features, tool_tensor)).item()
            orchestrator_score = self.orchestrator(dialog_features, ad_features, tool_tensor).item()
            score = 0.65 * orchestrator_score + 0.35 * judge_score
            scored.append((score, ad))
        scored.sort(key=lambda row: row[0], reverse=True)
        slate = [ad for _, ad in scored[:top_k]]
        return gate_prob, slate

    @torch.no_grad()
    def reward(self, example, slate):
        return slate_reward(example, slate)

    @torch.no_grad()
    def slate_relevance(self, example, slate):
        if not slate:
            return 0.0
        return sum(relevance_label(example, ad) for ad in slate) / len(slate)
