from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    state: torch.Tensor  # [D]
    action: torch.Tensor  # scalar
    episode_id: torch.Tensor  # scalar


class ToyAgentEpisodeDataset(Dataset):
    """Toy episodic dataset for agent memory / experience replay.

    Paper: APEX-EM (arXiv:2603.29093)

    We generate short episodes; the "optimal" action is a deterministic
    function of the state plus an episode-specific bias.
    """

    def __init__(
        self,
        *,
        num_episodes: int = 200,
        steps_per_episode: int = 20,
        state_dim: int = 64,
        num_actions: int = 12,
        seed: int = 0,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions

        rng = random.Random(seed)
        # A fixed linear mapping for the environment.
        w = [[rng.uniform(-1, 1) for _ in range(state_dim)] for _ in range(num_actions)]
        self._w = torch.tensor(w, dtype=torch.float32)
        self._episode_bias = torch.tensor([rng.uniform(-0.5, 0.5) for _ in range(num_episodes)], dtype=torch.float32)

        self._states: list[list[float]] = []
        self._actions: list[int] = []
        self._episode_ids: list[int] = []

        for ep in range(num_episodes):
            for _ in range(steps_per_episode):
                state = [rng.uniform(-1, 1) for _ in range(state_dim)]
                s = torch.tensor(state, dtype=torch.float32)
                logits = (self._w @ s) + self._episode_bias[ep]
                action = int(torch.argmax(logits).item())

                self._states.append(state)
                self._actions.append(action)
                self._episode_ids.append(ep)

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, idx: int) -> Sample:
        return Sample(
            state=torch.tensor(self._states[idx], dtype=torch.float32),
            action=torch.tensor(self._actions[idx], dtype=torch.long),
            episode_id=torch.tensor(self._episode_ids[idx], dtype=torch.long),
        )
