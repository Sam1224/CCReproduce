from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class PolicyOutput:
    decision_logits: torch.Tensor  # (B, 2) => [ABSTAIN, GENERATE]
    hint_logits: torch.Tensor  # (B, num_tools)


class MemPiPolicy(nn.Module):
    """A tiny parametric memory policy.

    This is a toy reproduction of the paper's key abstraction: a separate policy
    (pi_mem) that decides *when* to generate memory and *what* memory to generate.

    Here, memory content is discretized as a tool-id hint.
    """

    def __init__(self, *, vocab_size: int, num_tools: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.num_tools = num_tools
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decision_head = nn.Linear(hidden_dim, 2)
        self.hint_head = nn.Linear(hidden_dim, num_tools)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> PolicyOutput:
        # token_ids: (B, T)
        # mask: (B, T) boolean; True for real tokens
        emb = self.embed(token_ids)  # (B,T,D)
        mask_f = mask.to(emb.dtype).unsqueeze(-1)
        pooled = (emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        h = self.encoder(pooled)
        return PolicyOutput(decision_logits=self.decision_head(h), hint_logits=self.hint_head(h))

    @torch.no_grad()
    def sample_generate_hint(self, hint_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a hint id and return (hint_id, log_prob)."""

        probs = torch.softmax(hint_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        hint_id = dist.sample()
        log_prob = dist.log_prob(hint_id)
        return hint_id, log_prob

    @torch.no_grad()
    def generate_prob(self, decision_logits: torch.Tensor) -> torch.Tensor:
        # returns P(GENERATE)
        probs = torch.softmax(decision_logits, dim=-1)
        return probs[:, 1]
