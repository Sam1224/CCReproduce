"""E-VAds-R1: MLLM fine-tuned with RL for commercial intent reasoning.

Paper §4: E-VAds-R1 achieves 109.2% improvement in commercial intent reasoning
with only a few hundred training samples via reinforcement learning fine-tuning.

Key design:
  - Base: MLLM with video understanding capability
  - RL objective: maximize correctness on commercial intent Q&A
  - Reward: binary (correct/incorrect answer) or soft reward (semantic similarity)
  - Algorithm: GRPO (Group Relative Policy Optimization) or PPO

This toy implementation uses a lightweight text-based model to simulate the
E-VAds-R1 training pipeline. Replace with actual MLLM for production use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ToyMLLM(nn.Module):
    """Lightweight toy MLLM for E-VAds-R1 (simulates commercial intent reasoning).

    Real E-VAds-R1 uses a full MLLM (e.g., InternVL, LLaVA) with video tokens.
    This toy model processes tokenized question+description → answer embedding.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_answer_classes: int = 20,  # simplified answer classification
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_answer_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.embedding(input_ids)  # (B, L, D)
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0))
        # Use [CLS]-like pooling: mean over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(x)  # (B, num_classes)


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for E-VAds-R1.

    Paper uses GRPO/PPO to fine-tune MLLM with binary correctness reward.

    GRPO loss (simplified):
        L = -E[r(a) * log π(a|s)] + β * KL[π || π_ref]

    where:
        r(a) = 1 if answer is correct, 0 otherwise (normalized within group)
        π = current policy (MLLM)
        π_ref = reference policy (base MLLM before fine-tuning)
        β = KL penalty coefficient
    """

    def __init__(
        self,
        model: ToyMLLM,
        ref_model: ToyMLLM,
        lr: float = 1e-4,
        beta_kl: float = 0.1,
        group_size: int = 4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.beta_kl = beta_kl
        self.group_size = group_size
        self.device = device

    def compute_reward(
        self, pred_logits: Tensor, label_ids: Tensor
    ) -> Tensor:
        """Binary correctness reward: 1 if argmax matches label, else 0."""
        preds = pred_logits.argmax(dim=-1)
        return (preds == label_ids).float()

    def grpo_step(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        label_ids: Tensor,
    ) -> dict:
        """One GRPO optimization step.

        Args:
            input_ids: (B, L) tokenized input
            attention_mask: (B, L) mask
            label_ids: (B,) ground truth answer class

        Returns:
            dict with loss components
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        label_ids = label_ids.to(self.device)

        # Forward pass
        logits = self.model(input_ids, attention_mask)  # (B, C)

        # Compute reward
        with torch.no_grad():
            rewards = self.compute_reward(logits.detach(), label_ids)
            # Normalize rewards within group (GRPO)
            reward_mean = rewards.mean()
            reward_std = rewards.std().clamp(min=1e-8)
            normalized_rewards = (rewards - reward_mean) / reward_std

        # Policy gradient loss (REINFORCE-style)
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, label_ids.unsqueeze(1)).squeeze(1)
        pg_loss = -(normalized_rewards * chosen_log_probs).mean()

        # KL penalty against reference model
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids, attention_mask)
        ref_probs = F.softmax(ref_logits, dim=-1)
        curr_probs = F.softmax(logits, dim=-1)
        kl_div = F.kl_div(curr_probs.log(), ref_probs, reduction="batchmean")

        # Total GRPO loss
        loss = pg_loss + self.beta_kl * kl_div

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": rewards.mean().item(),
        }
