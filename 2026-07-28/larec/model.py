from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass
class LaRecConfig:
    num_items: int
    hidden_dim: int = 48
    reasoning_steps: int = 3
    seed_std: float = 0.15


class LaRecModel(nn.Module):
    def __init__(self, config: LaRecConfig):
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items, config.hidden_dim, padding_idx=0)
        self.history_encoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.seed_projector = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.reasoning_cell = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.state_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.reward_head = nn.Linear(config.hidden_dim, 1)

    def encode_history(self, history_item_ids: torch.Tensor) -> torch.Tensor:
        history_embeddings = self.item_embedding(history_item_ids)
        _, hidden = self.history_encoder(history_embeddings)
        return hidden.squeeze(0)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids)

    def build_personalized_seed(
        self,
        history_state: torch.Tensor,
        interest_item_ids: torch.Tensor,
        interest_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        interest_embeddings = self.item_embedding(interest_item_ids)
        normalized_mask = interest_mask / interest_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        user_interest = (interest_embeddings * normalized_mask.unsqueeze(-1)).sum(dim=1)
        seed_mean = self.seed_projector(torch.cat([history_state, user_interest], dim=-1))
        if deterministic:
            sampled_seed = seed_mean
        else:
            sampled_seed = seed_mean + torch.randn_like(seed_mean) * self.config.seed_std
        return sampled_seed, seed_mean

    def latent_reasoning(self, history_state: torch.Tensor, sampled_seed: torch.Tensor) -> torch.Tensor:
        state = sampled_seed
        states = []
        for _ in range(self.config.reasoning_steps):
            state = self.reasoning_cell(history_state, state)
            state = self.state_projector(state)
            states.append(state)
        return torch.stack(states, dim=1)

    def candidate_scores(self, final_state: torch.Tensor, target_item_id: torch.Tensor, negative_item_ids: torch.Tensor) -> torch.Tensor:
        target_embedding = self.item_embedding(target_item_id).unsqueeze(1)
        negative_embeddings = self.item_embedding(negative_item_ids)
        candidate_embeddings = torch.cat([target_embedding, negative_embeddings], dim=1)
        return torch.einsum("bd,bnd->bn", final_state, candidate_embeddings)

    def forward_pretrain(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        history_state = self.encode_history(batch["history_item_ids"])
        sampled_seed, _ = self.build_personalized_seed(
            history_state,
            batch["interest_item_ids"],
            batch["interest_mask"],
            deterministic=False,
        )
        latent_states = self.latent_reasoning(history_state, sampled_seed)
        step_embeddings = self.item_embedding(batch["step_item_ids"])
        target_embedding = self.item_embedding(batch["target_item_id"])

        step_alignment = functional.mse_loss(latent_states, step_embeddings)
        state_deltas = latent_states[:, 1:, :] - latent_states[:, :-1, :]
        target_direction = target_embedding.unsqueeze(1) - step_embeddings[:, :-1, :]
        process_direction = 1 - functional.cosine_similarity(state_deltas, target_direction, dim=-1).mean()

        final_state = latent_states[:, -1, :]
        scores = self.candidate_scores(final_state, batch["target_item_id"], batch["negative_item_ids"])
        rank_loss = functional.cross_entropy(scores, torch.zeros(scores.size(0), dtype=torch.long, device=scores.device))
        total_loss = rank_loss + 0.7 * step_alignment + 0.5 * process_direction

        return {
            "loss": total_loss,
            "rank_loss": rank_loss.detach(),
            "step_alignment": step_alignment.detach(),
            "process_direction": process_direction.detach(),
        }

    def forward_rl(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        history_state = self.encode_history(batch["history_item_ids"])
        sampled_seed, seed_mean = self.build_personalized_seed(
            history_state,
            batch["interest_item_ids"],
            batch["interest_mask"],
            deterministic=False,
        )
        latent_states = self.latent_reasoning(history_state, sampled_seed)
        final_state = latent_states[:, -1, :]
        target_embedding = self.item_embedding(batch["target_item_id"])
        scores = self.candidate_scores(final_state, batch["target_item_id"], batch["negative_item_ids"])
        log_prob = functional.log_softmax(scores, dim=-1)[:, 0]

        hit_reward = torch.sigmoid(scores[:, 0] - scores[:, 1:].mean(dim=1))
        semantic_reward = functional.cosine_similarity(final_state, target_embedding, dim=-1)
        reward = 0.6 * hit_reward + 0.4 * semantic_reward
        baseline = reward.detach().mean()
        policy_loss = -((reward.detach() - baseline) * log_prob).mean()
        regularization = functional.mse_loss(sampled_seed, seed_mean)
        supervised = functional.cross_entropy(scores, torch.zeros(scores.size(0), dtype=torch.long, device=scores.device))
        total_loss = policy_loss + 0.4 * supervised + 0.1 * regularization

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "reward": reward.detach().mean(),
            "regularization": regularization.detach(),
        }

    def recommend(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        history_state = self.encode_history(batch["history_item_ids"])
        sampled_seed, _ = self.build_personalized_seed(
            history_state,
            batch["interest_item_ids"],
            batch["interest_mask"],
            deterministic=True,
        )
        latent_states = self.latent_reasoning(history_state, sampled_seed)
        final_state = latent_states[:, -1, :]
        all_items = self.item_embedding.weight
        scores = final_state @ all_items.t()
        scores[:, 0] = -1e9

        history_item_ids = batch["history_item_ids"]
        batch_indices = torch.arange(scores.size(0), device=scores.device).unsqueeze(1)
        scores[batch_indices, history_item_ids] = -1e9
        return scores
