"""
GRPO (Group Relative Policy Optimization) trainer for QueryAgent-R1.
Implements the RL fine-tuning stage after SFT warm-up.

GRPO reward = consistency_reward (retrieval-purchase overlap) + ctr_reward (click-through proxy)
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Tuple
from model import QueryAgentR1, QueryAgentConfig


def compute_grpo_advantage(rewards: torch.Tensor, group_size: int = 4) -> torch.Tensor:
    """
    GRPO advantage: normalize rewards within each group of candidates.
    Group Relative Policy Optimization (DeepSeek-R1 style).

    rewards: [B * G] where G = group_size
    Returns: [B * G] advantage
    """
    B_G = rewards.size(0)
    assert B_G % group_size == 0
    rewards = rewards.view(-1, group_size)  # [B, G]
    mean = rewards.mean(dim=-1, keepdim=True)  # [B, 1]
    std = rewards.std(dim=-1, keepdim=True) + 1e-8  # [B, 1]
    advantage = (rewards - mean) / std  # [B, G]
    return advantage.view(-1)  # [B*G]


def grpo_policy_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_coeff: float = 0.01,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    Clipped policy gradient + KL penalty (PPO-style with GRPO advantages).
    log_probs, ref_log_probs: [B*G, T] (per-token log-probs of generated sequences)
    advantages: [B*G]
    """
    # Sequence-level log-prob
    seq_log_probs = log_probs.sum(dim=-1)          # [B*G]
    seq_ref_log_probs = ref_log_probs.sum(dim=-1)  # [B*G]

    ratio = (seq_log_probs - seq_ref_log_probs).exp()  # [B*G]
    clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # KL regularization: KL(policy || ref) ≈ ratio - 1 - log(ratio)
    kl = (ratio - 1 - ratio.log()).mean()

    return pg_loss + kl_coeff * kl


class GRPOTrainer:
    """
    Trains QueryAgent-R1 with GRPO.

    Pipeline per step:
    1. Generate G query candidates per user using the current policy.
    2. For each candidate: run chain-of-retrieval to get top-k products.
    3. Compute consistency_reward + ctr_reward.
    4. Compute GRPO advantages within each group.
    5. Update policy with clipped PG + KL penalty.
    """

    def __init__(
        self,
        model: QueryAgentR1,
        ref_model: QueryAgentR1,
        lr: float = 1e-5,
        group_size: int = 4,
        kl_coeff: float = 0.01,
        consistency_lambda: float = 0.5,
    ):
        self.model = model
        self.ref_model = ref_model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.consistency_lambda = consistency_lambda

    def compute_log_probs(
        self, model: QueryAgentR1, context: torch.Tensor, query_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log-probs for query_ids under the model."""
        logits = model.query_generator(context, query_ids[:, :-1])  # [B, T-1, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T-1, V]
        target = query_ids[:, 1:]  # [B, T-1]
        token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        # Mask padding
        mask = (target != 0).float()
        return (token_lp * mask)  # [B, T-1]

    def step(
        self,
        item_ids: torch.Tensor,
        behavior_types: torch.Tensor,
        attention_mask: torch.Tensor,
        product_catalog_emb: torch.Tensor,
        purchased_ids: torch.Tensor,
        ctr_labels: torch.Tensor,  # [B*G] binary CTR label for each candidate
    ) -> dict:
        """One GRPO training step."""
        B = item_ids.size(0)
        G = self.group_size

        # Encode user context (shared across groups)
        user_repr = self.model.user_encoder(item_ids, behavior_types, attention_mask)
        mem_repr = self.model.memory_module(user_repr)
        context = self.model.context_fuse(
            torch.cat([user_repr, mem_repr], dim=-1)
        )  # [B, H]

        # Expand context for G candidates
        context_expanded = context.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)  # [B*G, H]

        # Generate G query candidates per user
        with torch.no_grad():
            gen_queries = self.model.query_generator.generate(context_expanded)  # [B*G, L]

        # Chain-of-retrieval for each candidate
        query_emb = self.model.query_generator.token_emb(gen_queries).mean(dim=1)  # [B*G, H]
        _, retrieved_ids = self.model.product_retriever.retrieve(
            query_emb, product_catalog_emb, self.model.config.retrieval_top_k
        )

        # Purchased IDs expanded for B*G
        purchased_expanded = purchased_ids.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)

        # Consistency reward
        c_reward = self.model.consistency_reward(retrieved_ids, purchased_expanded)  # [B*G]

        # CTR reward (from online signal proxy or offline labels)
        ctr_reward = ctr_labels.float()  # [B*G]

        # Combined reward
        reward = (1 - self.consistency_lambda) * ctr_reward + self.consistency_lambda * c_reward

        # GRPO advantage
        advantage = compute_grpo_advantage(reward, G)  # [B*G]

        # Policy log-probs
        log_probs = self.compute_log_probs(self.model, context_expanded, gen_queries)  # [B*G, T-1]

        with torch.no_grad():
            ref_context_expanded = context_expanded.detach()
            ref_log_probs = self.compute_log_probs(
                self.ref_model, ref_context_expanded, gen_queries
            )  # [B*G, T-1]

        # GRPO loss
        loss = grpo_policy_loss(log_probs, ref_log_probs, advantage, self.kl_coeff)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward_mean": reward.mean().item(),
            "consistency_reward": c_reward.mean().item(),
            "ctr_reward": ctr_reward.mean().item(),
        }
