from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn

from dataset import DOMAINS, MECHANISMS, OBJECTIVES, RequestCase, SCENARIOS, STAGES
from evolution import StrategyConfig

MECHANISM_TO_ID = {name: index for index, name in enumerate(MECHANISMS)}
DOMAIN_TO_ID = {name: index for index, name in enumerate(DOMAINS)}
SCENARIO_TO_ID = {name: index for index, name in enumerate(SCENARIOS)}
STAGE_TO_ID = {name: index for index, name in enumerate(STAGES)}
OBJECTIVE_TO_ID = {name: index for index, name in enumerate(OBJECTIVES)}


@dataclass(frozen=True)
class ValueNetConfig:
    hidden_dim: int = 64
    embed_dim: int = 16
    metric_dim: int = 4


class ContextEncoder(nn.Module):
    def __init__(self, cfg: ValueNetConfig) -> None:
        super().__init__()
        self.domain_emb = nn.Embedding(len(DOMAINS), cfg.embed_dim)
        self.scenario_emb = nn.Embedding(len(SCENARIOS), cfg.embed_dim)
        self.stage_emb = nn.Embedding(len(STAGES), cfg.embed_dim)
        self.objective_emb = nn.Embedding(len(OBJECTIVES), cfg.embed_dim)
        self.weight_proj = nn.Sequential(
            nn.Linear(5, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )

    def forward(self, request_tensor: Dict[str, torch.Tensor]) -> torch.Tensor:
        structured = torch.cat(
            [
                self.domain_emb(request_tensor["domain_id"]),
                self.scenario_emb(request_tensor["scenario_id"]),
                self.stage_emb(request_tensor["stage_id"]),
                self.objective_emb(request_tensor["objective_id"]),
            ],
            dim=-1,
        )
        weights = self.weight_proj(request_tensor["weight_vector"])
        return torch.cat([structured, weights], dim=-1)


class StrategyValueNet(nn.Module):
    def __init__(self, cfg: ValueNetConfig) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(cfg)
        context_dim = cfg.embed_dim * 4 + cfg.hidden_dim
        self.mechanism_emb = nn.Embedding(len(MECHANISMS), cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + cfg.embed_dim + 4, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.metric_dim),
        )

    def forward(self, request_tensor: Dict[str, torch.Tensor], mechanism_id: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        context = self.context_encoder(request_tensor)
        mechanism = self.mechanism_emb(mechanism_id)
        features = torch.cat([context, mechanism, params], dim=-1)
        return self.mlp(features)


class StrategyGenerator:
    def __init__(self, value_net: StrategyValueNet, device: torch.device) -> None:
        self.value_net = value_net
        self.device = device

    def propose(self, request: RequestCase, ranked_candidates: Iterable[dict]) -> StrategyConfig:
        candidates = list(ranked_candidates)
        top_candidates = candidates[: min(3, len(candidates))]
        mechanism_votes: Dict[str, float] = {}
        for candidate in top_candidates:
            mechanism_votes[candidate["chunk"].mechanism] = mechanism_votes.get(candidate["chunk"].mechanism, 0.0) + float(candidate["score"])
        mechanism = max(mechanism_votes, key=mechanism_votes.get)
        params = [0.0, 0.0, 0.0, 0.0]
        weight_sum = 1e-6
        for candidate in top_candidates:
            if candidate["chunk"].mechanism != mechanism:
                continue
            weight = max(0.05, float(candidate["score"]))
            for index, value in enumerate(candidate["chunk"].params):
                params[index] += weight * value
            weight_sum += weight
        params = [value / weight_sum for value in params]
        return StrategyConfig(mechanism=mechanism, params=params)

    def refine(self, request: RequestCase, strategy: StrategyConfig, steps: int = 20, lr: float = 0.05) -> StrategyConfig:
        params = torch.tensor([strategy.params], dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=lr)
        request_tensor = request_to_tensor([request], self.device)
        mechanism_id = torch.tensor([MECHANISM_TO_ID[strategy.mechanism]], dtype=torch.long, device=self.device)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            predicted = self.value_net(request_tensor, mechanism_id, params)
            utility = predicted[:, 0] * request_tensor["core_weights"][:, 0] + predicted[:, 1] * request_tensor["core_weights"][:, 1]
            guardrail = request_tensor["guard_weights"] * torch.relu(-predicted[:, 2:4])
            objective = utility - request_tensor["lambda_penalty"] * guardrail.sum(dim=-1)
            loss = -objective.mean()
            loss.backward()
            optimizer.step()
            params.data.clamp_(0.0, 1.0)
        refined = params.detach().cpu().squeeze(0).tolist()
        return StrategyConfig(mechanism=strategy.mechanism, params=[float(value) for value in refined])


def request_to_tensor(requests: List[RequestCase], device: torch.device) -> Dict[str, torch.Tensor]:
    weight_vector = []
    core_weights = []
    guard_weights = []
    for request in requests:
        weight_vector.append(
            [
                request.core_weights["gmv"],
                request.core_weights["ctr"],
                request.guardrail_weights["refund_safety"],
                request.guardrail_weights["creator_fairness"],
                request.lambda_penalty,
            ]
        )
        core_weights.append([request.core_weights["gmv"], request.core_weights["ctr"]])
        guard_weights.append([request.guardrail_weights["refund_safety"], request.guardrail_weights["creator_fairness"]])
    return {
        "domain_id": torch.tensor([DOMAIN_TO_ID[request.domain] for request in requests], dtype=torch.long, device=device),
        "scenario_id": torch.tensor([SCENARIO_TO_ID[request.scenario] for request in requests], dtype=torch.long, device=device),
        "stage_id": torch.tensor([STAGE_TO_ID[request.stage] for request in requests], dtype=torch.long, device=device),
        "objective_id": torch.tensor([OBJECTIVE_TO_ID[request.objective] for request in requests], dtype=torch.long, device=device),
        "weight_vector": torch.tensor(weight_vector, dtype=torch.float32, device=device),
        "core_weights": torch.tensor(core_weights, dtype=torch.float32, device=device),
        "guard_weights": torch.tensor(guard_weights, dtype=torch.float32, device=device),
        "lambda_penalty": torch.tensor([[request.lambda_penalty] for request in requests], dtype=torch.float32, device=device),
    }
