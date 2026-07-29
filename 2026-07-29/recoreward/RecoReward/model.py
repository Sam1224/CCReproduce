from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentPolicy(nn.Module):
    def __init__(self, content_dim: int = 48, embed_dim: int = 32, hidden_dim: int = 128, num_fields: int = 6, vocab_size: int = 96):
        super().__init__()
        self.num_fields = num_fields
        self.vocab_size = vocab_size
        self.backbone = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.token_head = nn.Linear(hidden_dim, num_fields * vocab_size)
        self.description_proj = nn.Sequential(
            nn.Linear(num_fields * vocab_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, content: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(content)
        logits = self.token_head(hidden).view(content.size(0), self.num_fields, self.vocab_size)
        probs = F.softmax(logits, dim=-1)
        soft_bow = probs.flatten(1)
        description = F.normalize(self.description_proj(soft_bow), dim=-1)
        return {"logits": logits, "description": description}

    def sample_descriptions(self, content: torch.Tensor, rollouts: int = 8, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        logits = self.forward(content)["logits"] / temperature
        batch, fields, vocab = logits.shape
        expanded = logits.unsqueeze(1).expand(batch, rollouts, fields, vocab)
        distribution = torch.distributions.Categorical(logits=expanded)
        tokens = distribution.sample()
        logprob = distribution.log_prob(tokens).sum(dim=-1)
        one_hot = F.one_hot(tokens, vocab).float().flatten(2)
        description = F.normalize(self.description_proj(one_hot), dim=-1)
        return {"tokens": tokens, "logprob": logprob, "description": description}


class FrozenTwoTowerScorer(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.07), requires_grad=False)

    def score(self, users: torch.Tensor, descriptions: torch.Tensor) -> torch.Tensor:
        users = F.normalize(users, dim=-1)
        descriptions = F.normalize(descriptions, dim=-1)
        return torch.einsum("bmf,bgf->bgm", users, descriptions).mean(dim=-1)


def recommender_affinity_reward(descriptions: torch.Tensor, target_users: torch.Tensor, non_target_users: torch.Tensor, scorer: FrozenTwoTowerScorer, lambda_non_target: float = 2.0, format_reward: float = 1.0, alpha: float = 0.9) -> torch.Tensor:
    target_score = scorer.score(target_users, descriptions)
    non_target_score = scorer.score(non_target_users, descriptions)
    ras = target_score - lambda_non_target * non_target_score
    semantic_reward = torch.clamp((ras + (1.0 + lambda_non_target)) / (2.0 * (1.0 + lambda_non_target)), 0.0, 1.0)
    return alpha * semantic_reward + (1.0 - alpha) * format_reward


def group_relative_policy_loss(logprob: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    baseline = reward.mean(dim=1, keepdim=True)
    std = reward.std(dim=1, keepdim=True).clamp_min(1e-4)
    advantage = (reward - baseline) / std
    return -(logprob * advantage.detach()).mean()
