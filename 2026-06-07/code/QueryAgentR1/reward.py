"""
Consistency Reward for QueryAgent-R1 RL training.
Implements the joint CTR proxy + CVR proxy reward from §3.4 of arXiv:2606.05671.

The paper's key insight: standard query recommendation RL optimizes CTR proxy only,
leading to high click but low conversion. The consistency reward bridges the
query-product alignment gap by jointly rewarding:
  1. Query-user semantic relevance (CTR proxy)
  2. Retrieved-product–user preference alignment (CVR proxy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class CTRProxyReward(nn.Module):
    """
    Query-user semantic relevance reward (CTR proxy).
    Measures how well the generated query matches the user's interest embedding.

    Higher score = the query is semantically closer to the user's interests.
    """

    def __init__(self, query_encoder: nn.Module, hidden_dim: int):
        super().__init__()
        self.query_encoder = query_encoder  # encodes query text to embedding
        self.hidden_dim = hidden_dim

    def forward(
        self,
        query_embed: torch.Tensor,
        user_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_embed: (B, hidden_dim) — encoded generated query
            user_embed: (B, hidden_dim) — user interest embedding from memory module

        Returns:
            reward: (B,) — cosine similarity in [-1, 1], normalized to [0, 1]
        """
        q_norm = F.normalize(query_embed, dim=-1)
        u_norm = F.normalize(user_embed, dim=-1)
        cos_sim = (q_norm * u_norm).sum(dim=-1)  # (B,)
        return (cos_sim + 1.0) / 2.0  # map to [0, 1]


class CVRProxyReward(nn.Module):
    """
    Retrieved-product–user preference alignment reward (CVR proxy).
    Measures whether the products retrieved using the generated query
    actually match the user's downstream conversion preferences.

    In the paper, this is approximated using:
    - User's past conversion history as preference signal
    - Retrieved product set as the candidate pool
    - Overlap / ranking correlation as CVR proxy
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # MLP to predict user-product affinity score
        self.affinity_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        user_embed: torch.Tensor,
        retrieved_product_embeds: torch.Tensor,
        positive_product_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            user_embed: (B, hidden_dim)
            retrieved_product_embeds: (B, K, hidden_dim) — top-K retrieved products
            positive_product_embeds: (B, P, hidden_dim) — ground-truth converted products

        Returns:
            reward: (B,) — average affinity of retrieved products with user
        """
        B, K, D = retrieved_product_embeds.shape
        user_exp = user_embed.unsqueeze(1).expand(B, K, D)  # (B, K, D)

        pair_input = torch.cat([user_exp, retrieved_product_embeds], dim=-1)  # (B, K, 2D)
        affinity = self.affinity_scorer(pair_input).squeeze(-1)  # (B, K)

        if positive_product_embeds is not None:
            # Contrastive component: retrieved products should rank higher than random
            B2, P, D2 = positive_product_embeds.shape
            pos_user_exp = user_embed.unsqueeze(1).expand(B2, P, D2)
            pos_input = torch.cat([pos_user_exp, positive_product_embeds], dim=-1)
            pos_affinity = self.affinity_scorer(pos_input).squeeze(-1)  # (B, P)

            # Reward is higher when retrieved products have similar affinity to positives
            ret_mean = affinity.mean(dim=-1)  # (B,)
            pos_mean = pos_affinity.mean(dim=-1)  # (B,)
            reward = 1.0 - torch.clamp(pos_mean - ret_mean, min=0.0)
        else:
            reward = affinity.mean(dim=-1)  # (B,)

        return reward


class ConsistencyReward(nn.Module):
    """
    Joint Consistency Reward = α * CTR_proxy + β * CVR_proxy.

    This is the key RL training signal in QueryAgent-R1.
    α and β balance the two objectives; paper finds β helps close the
    query-level relevance vs. product-level conversion gap.
    """

    def __init__(
        self,
        query_encoder: nn.Module,
        hidden_dim: int,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        super().__init__()
        self.ctr_reward = CTRProxyReward(query_encoder, hidden_dim)
        self.cvr_reward = CVRProxyReward(hidden_dim)
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        query_embed: torch.Tensor,
        user_embed: torch.Tensor,
        retrieved_product_embeds: torch.Tensor,
        positive_product_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_embed: (B, hidden_dim)
            user_embed: (B, hidden_dim)
            retrieved_product_embeds: (B, K, hidden_dim)
            positive_product_embeds: (B, P, hidden_dim) optional ground-truth

        Returns:
            dict with keys: "total_reward", "ctr_reward", "cvr_reward"
        """
        ctr = self.ctr_reward(query_embed, user_embed)
        cvr = self.cvr_reward(user_embed, retrieved_product_embeds, positive_product_embeds)
        total = self.alpha * ctr + self.beta * cvr
        return {
            "total_reward": total,
            "ctr_reward": ctr,
            "cvr_reward": cvr,
        }


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) loss for RL fine-tuning.
    GRPO is used in the paper to optimize the query generator with the
    consistency reward.

    Reference: GRPO removes the value function by using group-relative
    advantage estimation: A_i = (r_i - mean(r)) / std(r).
    """

    def __init__(self, clip_epsilon: float = 0.2, kl_coef: float = 0.01):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            log_probs: (B,) — log prob of generated query under current policy
            old_log_probs: (B,) — log prob under old policy (before update)
            rewards: (B,) — consistency rewards
            ref_log_probs: (B,) — log prob under reference model (for KL penalty)

        Returns:
            dict with "loss", "policy_loss", "kl_loss"
        """
        # Group relative advantage: normalize rewards within the batch
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # PPO-clip objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # KL penalty from reference model (SFT model)
        kl_loss = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None:
            kl_loss = (torch.exp(log_probs) * (log_probs - ref_log_probs)).mean()

        total_loss = policy_loss + self.kl_coef * kl_loss

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
        }
