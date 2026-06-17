"""
Stage 1: Risk Filter (UNIVID).

Paper (§3.1): Lightweight binary classifier that pre-screens all incoming
segments to route high-risk content to the moderation actor.
Tuned for high recall to avoid missing violations (false negatives costly).

Architecture: shallow MLP on visual+text features, ~1M params.
Inference: ~2ms per segment, suitable for real-time pre-screening.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TextFeatureExtractor(nn.Module):
    """
    Simple text feature extractor using token embedding + mean pooling.
    In production: replaced by a frozen BERT/RoBERTa encoder.
    """

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        mask = (token_ids != 0).float().unsqueeze(-1)  # (B, L, 1)
        embeds = self.embedding(token_ids) * mask       # (B, L, E)
        mean_pool = embeds.sum(1) / mask.sum(1).clamp(min=1)  # (B, E)
        return self.proj(mean_pool)                             # (B, out_dim)


class RiskFilter(nn.Module):
    """
    Stage 1 lightweight risk pre-filter.

    Paper: "A lightweight filter removes obviously-safe segments
    before the heavier moderation actor runs."

    Fuses visual frame features + text features via a shallow MLP.
    Output: risk_score in [0, 1]; threshold ~0.3 for high-recall routing.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = visual_dim + text_dim

        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat: (B, visual_dim)
            text_feat:   (B, text_dim)
        Returns:
            risk_score: (B,) probabilities in [0, 1]
        """
        v = self.visual_norm(visual_feat)
        t = self.text_norm(text_feat)
        x = torch.cat([v, t], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))

    def route(self, risk_score: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
        """Returns boolean mask: True = route to moderation actor."""
        return risk_score > threshold


class RiskFilterLoss(nn.Module):
    """
    Binary cross-entropy with asymmetric weighting.
    Higher penalty for false negatives (missed violations).
    """

    def __init__(self, fn_weight: float = 3.0):
        super().__init__()
        self.fn_weight = fn_weight

    def forward(self, risk_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Upweight false negatives: violations that are missed
        pos_weight = torch.tensor(self.fn_weight, device=risk_scores.device)
        return F.binary_cross_entropy(
            risk_scores,
            labels.float(),
            weight=labels.float() * (self.fn_weight - 1) + 1,
        )


def simple_tokenize(
    texts,
    vocab_size: int = 10000,
    max_len: int = 64,
) -> torch.Tensor:
    """Hash-based tokenizer for toy experiments."""
    batch = []
    for text in texts:
        tokens = text.lower().split()[:max_len]
        ids = [hash(t) % (vocab_size - 1) + 1 for t in tokens]
        # Pad to max_len
        ids = ids + [0] * (max_len - len(ids))
        batch.append(ids)
    return torch.tensor(batch, dtype=torch.long)
