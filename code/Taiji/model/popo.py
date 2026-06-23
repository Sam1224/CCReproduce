"""
Pareto Optimal Policy Optimization (POPO) for Taiji.

POPO jointly optimizes two reward objectives:
  r1 = Semantic reward (LLM world knowledge quality)
  r2 = ID reward     (collaborative filtering alignment)

The key insight: naively combining r1 + r2 with fixed weights leads to
one objective dominating. POPO adaptively adjusts weights w1, w2 such
that the policy reaches a Pareto-optimal trade-off.

Pareto optimality condition:
  No further improvement in r1 is possible without degrading r2, and vice versa.

Algorithm (simplified from paper):
  1. Estimate current performance on both objectives: (v1_t, v2_t)
  2. Compute gradient direction for each objective: g1, g2
  3. Solve for Pareto-optimal combination weight λ* that minimizes
     the norm of the combined gradient (Frank-Wolfe update):
         λ* = argmin_{λ in Δ} || λ1*g1 + λ2*g2 ||^2
     This yields: λ* = clip(g2·(g2-g1) / ||g2-g1||^2, 0, 1)
  4. Combined gradient: g_combined = λ*·g1 + (1-λ*)·g2
  5. Policy update: θ ← θ - α * g_combined
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class POPOConfig:
    lr: float = 1e-5
    semantic_weight_init: float = 0.5   # initial λ for semantic reward
    id_weight_init: float = 0.5         # initial λ for ID reward
    clip_eps: float = 0.2               # PPO clip epsilon
    kl_coef: float = 0.01              # KL divergence penalty coefficient
    n_popo_steps: int = 10             # inner steps for weight adaptation


class SemanticReward(nn.Module):
    """
    Semantic reward: measures how well the LLM's reasoning aligns
    with world knowledge (proxy: cosine similarity to a reference embedding).
    In production Taiji: uses a pre-trained semantic reward model.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.ref_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, intent_embeddings: torch.Tensor, ref_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic reward as cosine similarity between
        model-generated intent and a reference embedding.

        Args:
            intent_embeddings: [B, D] from LLM enhancer
            ref_embeddings:    [B, D] reference semantic embedding

        Returns:
            reward: [B] in [-1, 1]
        """
        projected = self.ref_proj(intent_embeddings)
        cos_sim = nn.functional.cosine_similarity(projected, ref_embeddings, dim=-1)
        return cos_sim


class IDReward(nn.Module):
    """
    Collaborative ID reward: measures alignment between LLM's intent
    embedding and the collaborative filtering signal from user-item interactions.
    In production Taiji: uses online A/B CTR/CVR as signal.
    Proxy here: predicted CVR from the online ranker.
    """

    def forward(self, ctr_pred: torch.Tensor, cvr_pred: torch.Tensor) -> torch.Tensor:
        """
        Combine CTR and CVR predictions into ID reward.
        r_id = CTR * CVR  (IP: probability of impression → purchase)

        Args:
            ctr_pred: [B] in [0,1]
            cvr_pred: [B] in [0,1]

        Returns:
            reward: [B] in [0,1]
        """
        return ctr_pred * cvr_pred


class POPO(nn.Module):
    """
    Pareto Optimal Policy Optimization.

    Manages the adaptive trade-off between semantic and ID rewards
    during RL fine-tuning of the LLM policy.

    Core equation (Frank-Wolfe Pareto weight update):
        λ* = clip((g2 · (g2 - g1)) / ||g2 - g1||², 0, 1)
        g_combined = λ* · g1 + (1 - λ*) · g2
    """

    def __init__(self, config: POPOConfig):
        super().__init__()
        self.config = config
        # Current trade-off weight for semantic reward (1-λ for ID reward)
        self.register_buffer("semantic_weight", torch.tensor(config.semantic_weight_init))

    def compute_pareto_weight(
        self,
        grad_semantic: torch.Tensor,  # flattened gradient for semantic objective
        grad_id: torch.Tensor,        # flattened gradient for ID objective
    ) -> float:
        """
        Compute Frank-Wolfe Pareto-optimal weight λ* for semantic gradient.

        λ* = clip((g_id · (g_id - g_sem)) / ||g_id - g_sem||², 0, 1)

        This ensures the combined gradient direction is Pareto-improving:
        no objective is degraded while the other improves.
        """
        g1 = grad_semantic.float()
        g2 = grad_id.float()
        diff = g2 - g1
        denom = torch.dot(diff, diff)
        if denom < 1e-10:
            return 0.5  # degenerate case: gradients are the same
        numerator = torch.dot(g2, diff)
        lam = torch.clamp(numerator / denom, 0.0, 1.0)
        return lam.item()

    def popo_loss(
        self,
        log_probs: torch.Tensor,       # [B] current policy log probs
        old_log_probs: torch.Tensor,   # [B] old policy log probs (for PPO clip)
        semantic_rewards: torch.Tensor, # [B] r_sem
        id_rewards: torch.Tensor,       # [B] r_id
    ) -> dict:
        """
        Compute POPO loss = weighted combination of PPO losses for two objectives.

        PPO clip ratio: ρ = exp(log_π - log_π_old)

        For each objective k:
            L_k = -E[min(ρ * r_k, clip(ρ, 1-ε, 1+ε) * r_k)]

        POPO: L_combined = λ* * L_sem + (1-λ*) * L_id
        """
        # PPO probability ratio
        ratio = torch.exp(log_probs - old_log_probs)  # [B]

        # Normalize rewards
        def normalize(r):
            return (r - r.mean()) / (r.std() + 1e-8)

        r_sem = normalize(semantic_rewards)
        r_id = normalize(id_rewards)

        # PPO clipped surrogate for each objective
        def ppo_clip_loss(ratio, reward):
            clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
            loss = -torch.min(ratio * reward, clipped_ratio * reward)
            return loss.mean()

        loss_sem = ppo_clip_loss(ratio, r_sem)
        loss_id = ppo_clip_loss(ratio, r_id)

        # Use current Pareto weight (updated externally)
        lam = self.semantic_weight.item()
        combined_loss = lam * loss_sem + (1 - lam) * loss_id

        return {
            "loss": combined_loss,
            "loss_sem": loss_sem.item(),
            "loss_id": loss_id.item(),
            "semantic_weight": lam,
        }

    def update_weight(self, grad_semantic: torch.Tensor, grad_id: torch.Tensor):
        """Update the Pareto trade-off weight based on current gradients."""
        new_lam = self.compute_pareto_weight(grad_semantic, grad_id)
        self.semantic_weight.fill_(new_lam)
        return new_lam


class RejectionSampler:
    """
    Open-ended rejection sampling for high-quality CoT data generation.
    Used in Stage 1 (SFT) to filter the reverse-engineered CoT candidates.

    In Taiji: multiple CoT candidates are generated by the LLM, and only
    those with high semantic quality (measured by a reward model) are kept.
    """

    def __init__(self, quality_threshold: float = 0.7, n_candidates: int = 8):
        self.quality_threshold = quality_threshold
        self.n_candidates = n_candidates

    def score_cot(self, cot: str, user: dict) -> float:
        """
        Score CoT quality. In production: use a semantic reward model.
        Here: heuristic based on mention of key user attributes.
        """
        score = 0.0
        for cat in user.get("preferred_categories", []):
            if cat in cot:
                score += 0.3
        if user.get("preferred_price", "") in cot:
            score += 0.2
        if len(cot.split()) > 30:  # enough detail
            score += 0.3
        if "recommend" in cot.lower() or "intent" in cot.lower():
            score += 0.2
        return min(score, 1.0)

    def filter(self, cots: list[str], user: dict) -> Optional[str]:
        """
        Return the best-quality CoT that passes the threshold.
        Returns None if no candidate passes.
        """
        scored = [(cot, self.score_cot(cot, user)) for cot in cots]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_cot, best_score = scored[0]
        if best_score >= self.quality_threshold:
            return best_cot
        return None
