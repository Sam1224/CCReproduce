"""
QueryAgent-R1 Reward Functions

Two components:
1. CTR Reward:    estimated click-through probability for the query
2. CVR Consistency Reward: cosine similarity between mean retrieved product embedding
                           and the user preference vector

Joint reward:
    r = α · CTR_reward + (1-α) · CVR_consistency_reward

Paper section: "Consistency Reward in Chain-of-Retrieval RL"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CTREstimator(nn.Module):
    """
    Lightweight CTR predictor.
    Estimates the click probability of a (user, query) pair.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_emb: torch.Tensor, user_pref: torch.Tensor) -> torch.Tensor:
        # query_emb: (B, D), user_pref: (B, D)
        x = torch.cat([query_emb, user_pref], dim=-1)  # (B, 2D)
        return self.mlp(x).squeeze(-1)  # (B,)


class CVRConsistencyReward(nn.Module):
    """
    CVR Consistency Reward.
    Measures alignment between the products retrieved by a query
    and the user's purchase preference.

    Formula:
        R_CVR = cosine_sim( mean(product_embeddings), user_pref_embedding )

    Higher similarity → retrieved products match user preference → higher CVR alignment.
    """

    def forward(
        self,
        product_embeddings: torch.Tensor,  # (B, K, D)
        user_pref: torch.Tensor,           # (B, D)
    ) -> torch.Tensor:
        mean_product_emb = product_embeddings.mean(dim=1)  # (B, D)
        mean_product_emb = F.normalize(mean_product_emb, dim=-1)
        user_pref_norm = F.normalize(user_pref, dim=-1)
        return (mean_product_emb * user_pref_norm).sum(dim=-1)  # (B,) in [-1, 1]


class ConsistencyReward(nn.Module):
    """
    Combined reward module.

    r = alpha * CTR_reward + (1 - alpha) * CVR_consistency_reward

    where CVR_consistency_reward is normalized to [0, 1] via (x + 1) / 2.
    """

    def __init__(self, embed_dim: int = 64, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.ctr_estimator = CTREstimator(embed_dim)
        self.cvr_reward = CVRConsistencyReward()

    def forward(
        self,
        query_emb: torch.Tensor,          # (B, D) from retriever
        product_embeddings: torch.Tensor,  # (B, K, D) from retrieval
        user_pref: torch.Tensor,           # (B, D) from memory module
    ) -> Tuple[torch.Tensor, dict]:
        ctr = self.ctr_estimator(query_emb, user_pref)            # (B,) in [0, 1]
        cvr_raw = self.cvr_reward(product_embeddings, user_pref)  # (B,) in [-1, 1]
        cvr = (cvr_raw + 1.0) / 2.0                               # normalize to [0, 1]

        reward = self.alpha * ctr + (1.0 - self.alpha) * cvr      # (B,)

        return reward, {
            "ctr_reward": ctr,
            "cvr_reward": cvr,
            "total_reward": reward,
        }
