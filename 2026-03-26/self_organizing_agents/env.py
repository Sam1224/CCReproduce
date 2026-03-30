from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


TASK_TYPES = ["bugfix", "feature", "refactor"]


@dataclass
class Env:
    agent_skill: torch.Tensor  # (A, T) success probabilities


def make_env(seed: int = 1, agents: int = 5) -> Env:
    g = torch.Generator().manual_seed(seed)
    # Each agent has different success rate per task type.
    base = torch.rand(agents, len(TASK_TYPES), generator=g) * 0.4 + 0.5
    return Env(agent_skill=base)


def step(env: Env, task_type: int, agent_id: int, rng: torch.Generator) -> Tuple[float, float]:
    p = float(env.agent_skill[agent_id, task_type].item())
    ok = float((torch.rand((), generator=rng).item() < p))
    # reward combines quality and speed (toy).
    quality = ok
    speed = 1.0 - 0.5 * (1.0 - p)
    return ok, quality + 0.2 * speed
