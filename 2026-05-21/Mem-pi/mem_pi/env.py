from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import TOOL_NAMES, Task


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    chosen_tool_id: int


def heuristic_base_agent(task: Task, rng: np.random.Generator) -> int:
    """A simple downstream agent.

    If it sees an explicit token, it uses it. Otherwise it guesses.
    """

    explicit_prefix = "explicit:tool_"
    for tok in task.context_tokens:
        if tok.startswith(explicit_prefix):
            return int(tok[len(explicit_prefix) :])

    # Deterministic fallback: a weak agent that always defaults to tool 0.
    # This creates a clear regime where memory should abstain (when tool_id==0)
    # and generate (otherwise).
    return 0


def run_episode(
    *,
    task: Task,
    rng: np.random.Generator,
    hint_tool_id: Optional[int],
) -> EpisodeResult:
    if hint_tool_id is None:
        chosen = heuristic_base_agent(task, rng)
    else:
        chosen = int(hint_tool_id)

    return EpisodeResult(success=(chosen == task.tool_id), chosen_tool_id=chosen)
