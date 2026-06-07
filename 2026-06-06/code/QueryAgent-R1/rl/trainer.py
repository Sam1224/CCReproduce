"""
QueryAgent-R1 RL Training with REINFORCE / PPO-style gradient estimation.

Training loop:
    1. Sample user context from replay buffer
    2. Generate query tokens from QueryAgentR1
    3. Retrieve products via ProductRetriever
    4. Compute consistency reward (CTR + CVR)
    5. Update agent with policy gradient loss + baseline (REINFORCE)

For full PPO: replace REINFORCE with clipped surrogate objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict

from model.query_agent import QueryAgentR1
from retrieval.retriever import ProductRetriever
from rl.reward import ConsistencyReward


class REINFORCETrainer:
    """
    REINFORCE trainer with baseline for QueryAgent-R1.

    Policy gradient loss:
        L = -E[ (r - b) * log π(q | s) ]
    where b is a learned value baseline.
    """

    def __init__(
        self,
        agent: QueryAgentR1,
        retriever: ProductRetriever,
        reward_fn: ConsistencyReward,
        lr: float = 2e-5,
        gamma: float = 1.0,
        entropy_coeff: float = 0.01,
        device: str = "cpu",
    ):
        self.agent = agent.to(device)
        self.retriever = retriever.to(device)
        self.reward_fn = reward_fn.to(device)
        self.device = device
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        # Value baseline network
        self.value_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        self.optimizer = AdamW(
            list(agent.parameters()) + list(reward_fn.ctr_estimator.parameters())
            + list(self.value_net.parameters()),
            lr=lr, weight_decay=0.01,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        interaction_history = batch["interaction_history"].to(self.device)  # (B, M, D)
        current_context = batch["current_context"].to(self.device)          # (B, D)
        # gt_ctr: ground truth CTR labels (from logged data) not always available

        self.optimizer.zero_grad()

        # Forward: generate queries
        out = self.agent(interaction_history, current_context)
        user_pref = out["user_pref"]       # (B, D)
        query_logits = out["query_logits"] # (B, L, vocab)

        # Sample token actions from policy distribution
        probs = F.softmax(query_logits, dim=-1)          # (B, L, V)
        dist = torch.distributions.Categorical(probs=probs)
        query_ids = dist.sample()                        # (B, L)
        log_probs = dist.log_prob(query_ids).sum(dim=-1) # (B,) summed over tokens
        entropy = dist.entropy().sum(dim=-1)             # (B,)

        # Retrieve products for generated queries
        product_embs, _, query_emb = self.retriever(query_ids)  # (B, K, D), _, (B, D)

        # Compute reward
        reward, reward_info = self.reward_fn(query_emb, product_embs, user_pref)  # (B,)

        # Baseline
        baseline = self.value_net(user_pref).squeeze(-1)  # (B,)

        # Policy gradient loss
        advantage = (reward - baseline).detach()
        pg_loss = -(advantage * log_probs).mean()

        # Baseline MSE loss
        baseline_loss = F.mse_loss(baseline, reward.detach())

        # Entropy bonus (encourage exploration)
        entropy_loss = -self.entropy_coeff * entropy.mean()

        total_loss = pg_loss + baseline_loss + entropy_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.value_net.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "mean_reward": reward.mean().item(),
            "mean_ctr": reward_info["ctr_reward"].mean().item(),
            "mean_cvr": reward_info["cvr_reward"].mean().item(),
        }
