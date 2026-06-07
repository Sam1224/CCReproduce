"""
UNIVID Stage (A): Risk Filter
Fuses UNIVID caption embedding with raw modality signals to produce a risk score.
High-risk videos pass to the Moderation Actor; low-risk videos are cleared.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RiskFilter(nn.Module):
    """
    Multi-modal risk funnel.

    Inputs:
        caption_embedding: (B, embed_dim)   from UNIVID captioner
        visual_summary:    (B, vis_dim)     e.g., mean-pooled frame features
        audio_embedding:   (B, aud_dim)     e.g., speech transcript embedding

    Output:
        risk_score: (B,)  in [0, 1]
        is_high_risk: (B,) bool mask
    """

    def __init__(
        self,
        embed_dim: int = 256,
        vis_dim: int = 512,
        aud_dim: int = 128,
        hidden: int = 512,
        threshold: float = 0.3,
    ):
        super().__init__()
        self.threshold = threshold

        input_dim = embed_dim + vis_dim + aud_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        caption_embedding: torch.Tensor,
        visual_summary: torch.Tensor,
        audio_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([caption_embedding, visual_summary, audio_embedding], dim=-1)
        risk_score = self.mlp(fused).squeeze(-1)  # (B,)
        is_high_risk = risk_score >= self.threshold
        return risk_score, is_high_risk
