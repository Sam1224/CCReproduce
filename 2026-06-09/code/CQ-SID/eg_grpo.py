"""
EG-GRPO: Expert-Guided Group Relative Policy Optimization.

Aligns generative retrieval with downstream ranking goals.

Paper (Section 4):
  The policy pi_theta generates product CQ-SID codes given a query.
  Reward = expert ranker score for retrieved products.
  GRPO loss = -E[A * log pi_theta(codes|query)]
  where A = (r - mean(r)) / std(r)  [group-relative advantage]
  Expert guidance: uses a fixed expert model to define "good" retrievals.

This module implements:
1. Generative retriever (policy) that generates CQ-SID codes beam-search style
2. EG-GRPO training objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cq_sid import CQSIDEncoder


class GenerativeRetriever(nn.Module):
    """
    Generative retriever: given a query, generates K-level CQ-SID codes.
    Autoregressively predicts each level's code using the query and previous codes.

    Paper: generates product codes via beam search (simplified to greedy here).
    """

    def __init__(
        self,
        query_dim: int = 128,
        hidden_dim: int = 128,
        n_levels: int = 4,
        codebook_size: int = 256,
    ):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.code_embed = nn.Embedding(codebook_size, hidden_dim // n_levels)
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.codebook_size = codebook_size

        # Per-level prediction heads
        self.level_heads = nn.ModuleList([
            nn.Linear(hidden_dim + (hidden_dim // n_levels) * k, codebook_size)
            for k in range(n_levels)
        ])

    def forward(self, query_feat: torch.Tensor, target_codes: torch.Tensor = None):
        """
        Args:
            query_feat: (B, query_dim) query features
            target_codes: (B, K) target codes for teacher-forcing (training)
        Returns:
            logits_all: list of (B, codebook_size) logits per level
            sampled_codes: (B, K) sampled/greedy codes
        """
        B = query_feat.size(0)
        h = self.query_proj(query_feat)             # (B, hidden_dim)

        logits_all, sampled_codes = [], []
        prev_code_embeds = []

        for k, head in enumerate(self.level_heads):
            ctx = torch.cat([h] + prev_code_embeds, dim=-1)   # (B, hidden_dim + k*(hidden_dim//K))
            logits = head(ctx)                                  # (B, codebook_size)
            logits_all.append(logits)

            if target_codes is not None:
                # Teacher forcing: use ground truth code
                code_k = target_codes[:, k]
            else:
                # Greedy decoding
                code_k = logits.argmax(-1)

            sampled_codes.append(code_k)
            prev_code_embeds.append(self.code_embed(code_k))   # (B, hidden_dim//K)

        sampled_codes = torch.stack(sampled_codes, dim=1)       # (B, K)
        return logits_all, sampled_codes


class ExpertRanker(nn.Module):
    """
    Simulated expert ranker that scores (query, product_codes) pairs.
    In the paper, this is the downstream production ranker that provides reward.
    """

    def __init__(self, query_dim: int = 128, hidden_dim: int = 64, n_levels: int = 4, codebook_size: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.code_embed = nn.Embedding(codebook_size, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + n_levels * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.n_levels = n_levels

    @torch.no_grad()
    def score(self, query_feat: torch.Tensor, codes: torch.Tensor):
        """
        Args:
            query_feat: (B, query_dim)
            codes: (B, K) product codes
        Returns:
            scores: (B,) relevance scores
        """
        q = self.query_proj(query_feat)
        c = torch.cat([self.code_embed(codes[:, k]) for k in range(self.n_levels)], dim=-1)
        return self.scorer(torch.cat([q, c], dim=-1)).squeeze(-1)


def eg_grpo_loss(
    retriever: GenerativeRetriever,
    expert: ExpertRanker,
    query_feat: torch.Tensor,
    n_samples: int = 4,
    temperature: float = 1.0,
):
    """
    EG-GRPO loss: Group Relative Policy Optimization with expert reward.

    For each query, sample n_samples code sequences, compute expert reward,
    use group-relative advantage to compute policy gradient loss.

    Paper (Section 4.2):
        A_i = (r_i - mean(r)) / (std(r) + eps)   # group-relative advantage
        L_GRPO = -E[A_i * sum_k log pi(c_k | query, c_{<k})]
    """
    B = query_feat.size(0)
    device = query_feat.device

    all_rewards = []
    all_log_probs = []

    # Sample n_samples trajectories per query
    for _ in range(n_samples):
        logits_all, sampled_codes = retriever(query_feat)   # (B, K)

        # Compute log probabilities of sampled codes
        log_prob = torch.zeros(B, device=device)
        for k, logits in enumerate(logits_all):
            log_p_k = F.log_softmax(logits / temperature, dim=-1)
            log_prob += log_p_k.gather(1, sampled_codes[:, k:k+1]).squeeze(-1)

        # Expert reward (no gradient through expert)
        with torch.no_grad():
            reward = expert.score(query_feat, sampled_codes)

        all_rewards.append(reward)
        all_log_probs.append(log_prob)

    rewards = torch.stack(all_rewards, dim=1)       # (B, n_samples)
    log_probs = torch.stack(all_log_probs, dim=1)   # (B, n_samples)

    # Group-relative advantage normalization (per-query, across samples)
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards - mean_r) / std_r          # (B, n_samples)

    # Policy gradient loss: -E[A * log pi]
    loss = -(advantages * log_probs).mean()

    return loss, rewards.mean().item()
