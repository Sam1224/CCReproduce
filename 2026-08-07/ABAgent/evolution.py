from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dataset import RequestCase


@dataclass
class StrategyConfig:
    mechanism: str
    params: List[float]


@dataclass
class ExperimentNode:
    step: int
    strategy: StrategyConfig
    observed_metrics: Dict[str, float]
    utility: float
    parent_index: Optional[int]


class ExperimentTree:
    def __init__(self, root: ExperimentNode) -> None:
        self.nodes: List[ExperimentNode] = [root]

    def add(self, node: ExperimentNode) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def best(self) -> ExperimentNode:
        return max(self.nodes, key=lambda node: node.utility)


def compute_utility(request: RequestCase, observed_metrics: Dict[str, float]) -> float:
    discounted_metrics = {
        key: observed_metrics[key] * observed_metrics.get("confidence", 1.0)
        for key in ("gmv", "ctr", "refund_safety", "creator_fairness")
    }
    core_value = sum(request.core_weights.get(key, 0.0) * discounted_metrics[key] for key in request.core_weights)
    guardrail_penalty = 0.0
    for key, weight in request.guardrail_weights.items():
        guardrail_penalty += weight * max(0.0, -discounted_metrics[key])
    return round(core_value - request.lambda_penalty * guardrail_penalty, 4)
