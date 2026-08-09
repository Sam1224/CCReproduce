from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AgentOPSDConfig:
    vocab_size: int = 128
    num_actions: int = 6
    hidden_size: int = 96
    turn_gap_scale: float = 1.0
    eps: float = 1e-6


class TinyAgentPolicy(nn.Module):
    def __init__(self, config: AgentOPSDConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        self.action_head = nn.Linear(config.hidden_size, config.num_actions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, turns, tokens = observations.shape
        embedded = self.embedding(observations).mean(dim=2)
        hidden, _ = self.encoder(embedded)
        return self.action_head(hidden).view(batch_size, turns, self.config.num_actions)

    def action_log_probs(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self(observations)
        return F.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)


class AgentOPSDCredit(nn.Module):
    def __init__(self, config: Optional[AgentOPSDConfig] = None):
        super().__init__()
        self.config = config or AgentOPSDConfig()

    def forward(
        self,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        token_mask: torch.Tensor,
        rewards: torch.Tensor,
        group_success_rate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_tokens = token_mask.sum(dim=-1).clamp_min(1.0)
        token_gap = (teacher_log_probs - student_log_probs) * token_mask
        turn_evidence = token_gap.sum(dim=-1) / valid_tokens.sqrt()
        turn_evidence = turn_evidence * self.config.turn_gap_scale

        if group_success_rate is None:
            group_success_rate = rewards.float().mean().expand_as(rewards)
        group_success_rate = group_success_rate.float().clamp(self.config.eps, 1 - self.config.eps)
        initial_log_odds = torch.logit(group_success_rate).unsqueeze(-1)

        belief_log_odds = initial_log_odds + torch.cumsum(turn_evidence, dim=-1)
        beliefs = torch.sigmoid(belief_log_odds)
        beliefs_with_initial = torch.cat([group_success_rate.unsqueeze(-1), beliefs], dim=-1)
        marginal_revision = beliefs_with_initial[:, 1:] - beliefs_with_initial[:, :-1]

        outcome_direction = rewards.float().mul(2.0).sub(1.0).unsqueeze(-1)
        signed_credit = marginal_revision * outcome_direction
        credit_centered = signed_credit - signed_credit.mean(dim=-1, keepdim=True)
        normalizer = credit_centered.abs().sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        turn_credit = credit_centered / normalizer
        return turn_credit, beliefs


def weighted_policy_loss(log_probs: torch.Tensor, rewards: torch.Tensor, turn_credit: torch.Tensor) -> torch.Tensor:
    trajectory_advantage = rewards.float() - rewards.float().mean()
    advantages = trajectory_advantage.unsqueeze(-1) * turn_credit.detach()
    return -(advantages * log_probs).mean()
