from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Action = int


@dataclass
class GridTask:
    start: Tuple[int, int]
    goal: Tuple[int, int]
    horizon: int = 12


class MiniGridWorld:
    """A tiny deterministic grid world for long-horizon agent training.

    - State includes (x, y, goal_x, goal_y, t).
    - Actions: 0=up, 1=down, 2=left, 3=right.
    - Reward: sparse terminal reward (1 if reach goal within horizon else 0).

    This is intentionally minimal to serve as a runnable HINT-SD scaffold.
    """

    def __init__(self, size: int = 5):
        self.size = size

    def sample_task(self, rng: np.random.Generator, horizon: int = 12) -> GridTask:
        start = (int(rng.integers(0, self.size)), int(rng.integers(0, self.size)))
        goal = start
        while goal == start:
            goal = (int(rng.integers(0, self.size)), int(rng.integers(0, self.size)))
        return GridTask(start=start, goal=goal, horizon=horizon)

    def optimal_action(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> Action:
        x, y = pos
        gx, gy = goal
        if x < gx:
            return 3
        if x > gx:
            return 2
        if y < gy:
            return 1
        if y > gy:
            return 0
        return 0

    def step(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        x, y = pos
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        x = int(np.clip(x, 0, self.size - 1))
        y = int(np.clip(y, 0, self.size - 1))
        return (x, y)

    def rollout(self, task: GridTask, policy_fn) -> Tuple[List[np.ndarray], List[Action], int]:
        """Rollout an episode.

        Returns (states, actions, terminal_reward).
        """
        pos = task.start
        states: List[np.ndarray] = []
        actions: List[Action] = []
        for t in range(task.horizon):
            state = self.encode_state(pos, task.goal, t, task.horizon)
            action = int(policy_fn(state))
            states.append(state)
            actions.append(action)
            pos = self.step(pos, action)
            if pos == task.goal:
                return states, actions, 1
        return states, actions, 0

    def encode_state(self, pos: Tuple[int, int], goal: Tuple[int, int], t: int, horizon: int) -> np.ndarray:
        x, y = pos
        gx, gy = goal
        return np.array(
            [x / (self.size - 1), y / (self.size - 1), gx / (self.size - 1), gy / (self.size - 1), t / max(1, horizon - 1)],
            dtype=np.float32,
        )
