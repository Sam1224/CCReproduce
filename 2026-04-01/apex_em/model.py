from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MemoryQueryResult:
    neighbor_actions: torch.Tensor  # [B, K]
    neighbor_weights: torch.Tensor  # [B, K]


class EpisodicMemory:
    """A simple non-parametric memory over (state -> action).

    It stores state vectors and returns top-k neighbors by cosine similarity.
    """

    def __init__(self, *, state_dim: int, max_size: int = 20000) -> None:
        self.state_dim = state_dim
        self.max_size = max_size
        self._states: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None

    def build(self, *, states: torch.Tensor, actions: torch.Tensor) -> None:
        # states: [N,D], actions: [N]
        if states.shape[0] > self.max_size:
            states = states[: self.max_size]
            actions = actions[: self.max_size]
        self._states = torch.nn.functional.normalize(states, dim=-1)
        self._actions = actions

    def query(self, q: torch.Tensor, *, top_k: int = 8) -> MemoryQueryResult:
        if self._states is None or self._actions is None:
            raise RuntimeError("memory not built")

        qn = torch.nn.functional.normalize(q, dim=-1)
        sim = qn @ self._states.T  # [B,N]
        w, idx = torch.topk(sim, k=min(top_k, sim.shape[1]), dim=-1)
        neighbor_actions = self._actions[idx]
        neighbor_weights = torch.softmax(w, dim=-1)
        return MemoryQueryResult(neighbor_actions=neighbor_actions, neighbor_weights=neighbor_weights)


class APEXEMToyPolicy(nn.Module):
    """Parametric policy + non-parametric replay blending.

    The paper describes structured procedural-episodic replay; here we implement
    a minimal retrieval-augmented policy that blends:
    - an MLP policy over state
    - a kNN action distribution from episodic memory
    """

    def __init__(self, *, state_dim: int = 64, num_actions: int = 12) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.logit_head = nn.Linear(128, num_actions)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, *, state: torch.Tensor, memory_dist: torch.Tensor) -> torch.Tensor:
        # memory_dist: [B, A]
        h = self.encoder(state)
        logits = self.logit_head(h)
        gate = self.gate(h)  # [B,1]

        param_p = torch.softmax(logits, dim=-1)
        mix_p = gate * param_p + (1 - gate) * memory_dist
        # return log-probabilities for stable CE
        return torch.log(mix_p.clamp_min(1e-8))
