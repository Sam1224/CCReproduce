from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMemory(nn.Module):
    def __init__(self, input_size: int = 37, hidden_size: int = 96):
        super().__init__()
        self.working_memory = nn.GRUCell(input_size, hidden_size)
        self.episodic_memory = nn.GRUCell(hidden_size, hidden_size)
        self.preference_head = nn.Linear(hidden_size, 5)
        self.update_gate = nn.Linear(hidden_size + input_size, 1)

    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, steps, _ = observations.shape
        working = observations.new_zeros(batch_size, self.working_memory.hidden_size)
        episodic = observations.new_zeros(batch_size, self.episodic_memory.hidden_size)
        memories = []
        update_probs = []
        for step in range(steps):
            current = observations[:, step]
            working = self.working_memory(current, working)
            update_prob = torch.sigmoid(self.update_gate(torch.cat([working, current], dim=-1)))
            episodic_candidate = self.episodic_memory(working, episodic)
            episodic = update_prob * episodic_candidate + (1.0 - update_prob) * episodic
            memories.append(torch.cat([working, episodic], dim=-1))
            update_probs.append(update_prob)
        memory = torch.stack(memories, dim=1)
        preference_logits = self.preference_head(episodic)
        return {"memory": memory, "preference_logits": preference_logits, "update_probs": torch.stack(update_probs, dim=1)}


class RecVerseModel(nn.Module):
    def __init__(self, action_count: int = 9, hidden_size: int = 96):
        super().__init__()
        self.memory = HierarchicalMemory(hidden_size=hidden_size)
        self.policy = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Linear(hidden_size, action_count))
        self.value = nn.Linear(hidden_size * 2, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        observations = torch.cat([batch["screen"], batch["category"]], dim=-1)
        memory_outputs = self.memory(observations)
        logits = self.policy(memory_outputs["memory"])
        values = self.value(memory_outputs["memory"]).squeeze(-1)
        return {**memory_outputs, "action_logits": logits, "values": values}


def trajectory_reward(batch: Dict[str, torch.Tensor], sampled_actions: torch.Tensor) -> torch.Tensor:
    preferred_category = F.one_hot(batch["preference"], num_classes=batch["category"].size(-1)).float()
    seen_preferred = (batch["category"] * preferred_category.unsqueeze(1)).sum(dim=-1)
    active_actions = sampled_actions.ne(7).float().mean(dim=1)
    intent_actions = sampled_actions.eq(5).float() + sampled_actions.eq(6).float() + 0.5 * sampled_actions.eq(3).float()
    intent_score = (intent_actions * seen_preferred).mean(dim=1)
    macro_score = 1.0 - (active_actions - 0.7).abs()
    return intent_score + 0.5 * macro_score


def recverse_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    imitation = F.cross_entropy(outputs["action_logits"].flatten(0, 1), batch["actions"].flatten())
    preference = F.cross_entropy(outputs["preference_logits"], batch["preference"])
    action_distribution = torch.distributions.Categorical(logits=outputs["action_logits"])
    sampled_actions = action_distribution.sample()
    rewards = trajectory_reward(batch, sampled_actions)
    advantages = rewards.unsqueeze(1) - outputs["values"].detach()
    policy_loss = -(action_distribution.log_prob(sampled_actions) * advantages).mean()
    value_loss = F.mse_loss(outputs["values"], rewards.unsqueeze(1).expand_as(outputs["values"]))
    memory_sparsity = outputs["update_probs"].mean()
    return imitation + 0.5 * preference + 0.2 * policy_loss + 0.1 * value_loss + 0.01 * memory_sparsity
