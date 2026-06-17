"""
UNIVID-Lite: Lightweight moderation actor for standard violation cases.

Paper (§3.2): "UNIVID-Lite handles the majority of standard violations
with low latency. It operates on the policy-aware caption embedding
produced by the policy caption stage."

Architecture:
  - Inputs: visual features + policy caption embedding
  - 3-layer transformer encoder (much smaller than full VLM)
  - Head: binary violation + category classification
  - Latency: ~5ms per segment (production SLA)

This is the primary path in the Moderation Actor. Novel/ambiguous cases
that UNIVID-Lite is uncertain about are escalated to UNIVID-RAG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UniVIDLite(nn.Module):
    """
    Lightweight moderation model conditioned on policy-aware captions.

    Paper: "UNIVID-Lite receives the policy-aware caption as a soft prompt
    alongside multimodal features and produces moderation decisions."

    Uncertainty estimation: entropy of category distribution used to
    route uncertain cases to UNIVID-RAG.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        caption_dim: int = 512,
        hidden_dim: int = 512,
        num_categories: int = 8,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.caption_proj = nn.Linear(caption_dim, hidden_dim)
        self.type_emb = nn.Embedding(2, hidden_dim)  # 0=visual, 1=caption

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Dual head: binary + category
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.category_head = nn.Linear(hidden_dim, num_categories)

    def forward(
        self,
        visual_feat: torch.Tensor,
        caption_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feat:   (B, visual_dim)
            caption_embed: (B, caption_dim)
        Returns:
            binary_logits:   (B, 2)
            category_logits: (B, num_categories)
            feat:            (B, hidden_dim) for uncertainty routing
        """
        v = self.visual_proj(visual_feat).unsqueeze(1)       # (B, 1, H)
        c = self.caption_proj(caption_embed).unsqueeze(1)    # (B, 1, H)

        # Add type embeddings
        type_ids = torch.tensor([0, 1], device=visual_feat.device)
        type_embs = self.type_emb(type_ids).unsqueeze(0)    # (1, 2, H)

        tokens = torch.cat([v, c], dim=1) + type_embs       # (B, 2, H)
        out = self.transformer(tokens)                        # (B, 2, H)
        feat = self.norm(out.mean(1))                         # (B, H)

        return self.binary_head(feat), self.category_head(feat), feat

    def uncertainty(self, category_logits: torch.Tensor) -> torch.Tensor:
        """
        Entropy-based uncertainty estimate.
        High entropy → route to UNIVID-RAG.
        """
        probs = torch.softmax(category_logits, dim=-1)
        # Shannon entropy, normalized by log(num_classes)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        return entropy / torch.log(torch.tensor(float(probs.shape[-1])))


class UniVIDLiteLoss(nn.Module):
    """
    Combined loss for UNIVID-Lite.

        L = L_binary + λ_cat * L_category
    """

    def __init__(self, lambda_cat: float = 0.5):
        super().__init__()
        self.lambda_cat = lambda_cat

    def forward(
        self,
        binary_logits: torch.Tensor,
        category_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        category_labels: torch.Tensor,
    ) -> torch.Tensor:
        l_binary = F.cross_entropy(binary_logits, binary_labels)
        l_cat = F.cross_entropy(category_logits, category_labels)
        return l_binary + self.lambda_cat * l_cat
