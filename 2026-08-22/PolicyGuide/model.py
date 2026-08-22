from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class WorkflowState:
    completed_steps: List[str] = field(default_factory=list)
    open_requests: List[str] = field(default_factory=list)


class PolicyGraph:
    def __init__(self, steps: Iterable[str]):
        self.steps = list(steps)
        self.index = {step: pos for pos, step in enumerate(self.steps)}

    def first_unmet_step(self, completed_steps: Iterable[str]) -> str:
        completed = set(completed_steps)
        for step in self.steps:
            if step not in completed:
                return step
        return "ready_for_mutation"

    def allows(self, action: str, completed_steps: Iterable[str]) -> bool:
        if action not in self.index:
            return self.first_unmet_step(completed_steps) == "ready_for_mutation"
        expected = self.first_unmet_step(completed_steps)
        return action == expected


class ProactiveVerifier(nn.Module):
    def __init__(self, num_steps: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_steps),
        )

    def forward(self, completed_step_bitmap: torch.Tensor) -> torch.Tensor:
        return self.net(completed_step_bitmap.float())


def bitmap_for_steps(completed: Iterable[str], vocab: Dict[str, int]) -> torch.Tensor:
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    for step in completed:
        if step in vocab:
            vector[vocab[step]] = 1.0
    return vector


def compile_policy(required_steps: List[str]) -> PolicyGraph:
    return PolicyGraph(required_steps)


def guide_next_action(policy: PolicyGraph, state: WorkflowState, proposed_action: str) -> Tuple[bool, str]:
    unmet = policy.first_unmet_step(state.completed_steps)
    if unmet == "ready_for_mutation":
        return True, f"All prerequisites are satisfied; action `{proposed_action}` is allowed."
    if proposed_action != unmet:
        return False, f"Blocked `{proposed_action}`. Complete `{unmet}` before continuing."
    return True, f"Proceed with required step `{unmet}` and persist the resulting evidence."
