from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


ACTIONS = ["up", "down", "left", "right"]


@dataclass
class Transition:
    obs: torch.Tensor
    act: int


class GridEnv:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.reset()

    def reset(self) -> torch.Tensor:
        self.x = 0
        self.y = 0
        self.goal = (self.size - 1, self.size - 1)
        return self._obs()

    def _obs(self) -> torch.Tensor:
        # observation vector: (x,y,goalx,goaly)
        return torch.tensor([self.x, self.y, self.goal[0], self.goal[1]], dtype=torch.float32) / (self.size - 1)

    def step(self, a: int) -> Tuple[torch.Tensor, float, bool]:
        if a == 0 and self.y > 0:
            self.y -= 1
        elif a == 1 and self.y < self.size - 1:
            self.y += 1
        elif a == 2 and self.x > 0:
            self.x -= 1
        elif a == 3 and self.x < self.size - 1:
            self.x += 1

        done = (self.x, self.y) == self.goal
        reward = 1.0 if done else -0.01
        return self._obs(), reward, done


def expert_policy(obs: torch.Tensor) -> int:
    # greedy to goal
    x, y, gx, gy = obs * 4
    if gx > x:
        return 3
    if gy > y:
        return 1
    return 0


def build_offline_dataset(n: int = 2000, seed: int = 1) -> List[Transition]:
    g = torch.Generator().manual_seed(seed)
    data: List[Transition] = []
    env = GridEnv(size=5)

    for _ in range(n):
        obs = env.reset()
        # randomize start
        env.x = int(torch.randint(0, env.size, (1,), generator=g).item())
        env.y = int(torch.randint(0, env.size, (1,), generator=g).item())
        obs = env._obs()

        for _ in range(6):
            a = expert_policy(obs)
            data.append(Transition(obs=obs, act=a))
            obs, _, done = env.step(a)
            if done:
                break

    return data
