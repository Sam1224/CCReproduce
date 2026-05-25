from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from env import Action, GridTask, MiniGridWorld


@dataclass
class HindsightSignal:
    step_index: int
    feedback_action: Action


def select_failure_relevant_steps(
    env: MiniGridWorld,
    task: GridTask,
    states: List[np.ndarray],
    actions: List[Action],
    terminal_reward: int,
    max_steps: int = 1,
) -> List[HindsightSignal]:
    """Toy hindsight analyzer.

    The original HINT-SD uses full-trajectory hindsight with an LLM analyzer to pick
    failure-relevant action spans. In this toy setting we use an oracle: in failed
    trajectories, select the earliest step(s) where the action deviates from the
    greedy optimal action.

    Returns at most `max_steps` signals.
    """
    if terminal_reward == 1:
        return []

    pos = task.start
    signals: List[HindsightSignal] = []
    for t, action in enumerate(actions):
        optimal = env.optimal_action(pos, task.goal)
        if action != optimal:
            signals.append(HindsightSignal(step_index=t, feedback_action=optimal))
            if len(signals) >= max_steps:
                break
        pos = env.step(pos, action)

    if not signals:
        # fallback: supervise the last step with the current optimal move
        optimal = env.optimal_action(pos, task.goal)
        signals.append(HindsightSignal(step_index=len(actions) - 1, feedback_action=optimal))

    return signals
