"""
Policy-Aware Caption Generation (UNIVID core concept).

Paper (§3.2): "UNIVID generates interpretable policy-aligned captions as an
intermediate representation before making moderation decisions."

The key insight: instead of black-box classification, the VLM first produces
a textual explanation anchored to platform policy language, then the
moderation decision is conditioned on that explanation.

Architecture:
  - PolicyCaptionEncoder: encodes visual+text into a latent aligned with
    policy vocabulary (simulates fine-tuned VLM encoder)
  - PolicyCaptionDecoder: autoregressive MLP-based caption generator
    (production: full transformer LM fine-tuned on expert-labeled captions)
  - PolicyAlignedEmbedding: projects caption embedding into moderation space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class PolicyCaptionEncoder(nn.Module):
    """
    Encodes multimodal input into a policy-aware latent.

    Paper: VLM encoder fine-tuned on expert-annotated (segment, caption) pairs
    where captions explicitly reference platform policy categories.

    Simulated here as a cross-modal attention block over visual+text tokens.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 256,
        latent_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 2, latent_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: (B, visual_dim)
            text_feat:   (B, text_dim)
        Returns:
            latent: (B, latent_dim)
        """
        v = self.visual_proj(visual_feat).unsqueeze(1)  # (B, 1, D)
        t = self.text_proj(text_feat).unsqueeze(1)      # (B, 1, D)
        tokens = torch.cat([v, t], dim=1) + self.pos_emb  # (B, 2, D)
        out = self.transformer(tokens)                     # (B, 2, D)
        latent = self.out_norm(out.mean(1))                # (B, D) mean pooling
        return latent


class PolicyCaptionDecoder(nn.Module):
    """
    Decodes the policy-aware latent into a caption embedding.

    Production: autoregressive LM (GPT-style) fine-tuned on
    (segment_latent, policy_caption) pairs.

    Here: MLP projecting latent → caption embedding space.
    The caption embedding space is shared with a frozen caption encoder
    so we can compute similarity to retrieved historical captions (UNIVID-RAG).
    """

    def __init__(self, latent_dim: int = 512, caption_dim: int = 512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, caption_dim),
        )
        self.norm = nn.LayerNorm(caption_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim)
        Returns:
            caption_embed: (B, caption_dim) L2-normalized
        """
        out = self.decoder(latent)
        return F.normalize(self.norm(out), dim=-1)


class PolicyAlignedEmbedding(nn.Module):
    """
    Projects caption embedding + raw latent into violation decision space.

    This is the interpretable bridge: moderation decisions are conditioned
    on the policy-aware caption, not just raw visual features.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        caption_dim: int = 512,
        decision_dim: int = 256,
        num_categories: int = 8,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + caption_dim, decision_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decision_dim * 2, decision_dim),
        )
        self.classifier = nn.Linear(decision_dim, num_categories)

    def forward(
        self,
        latent: torch.Tensor,
        caption_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent:        (B, latent_dim)
            caption_embed: (B, caption_dim)
        Returns:
            decision_feat: (B, decision_dim) for downstream use
            category_logits: (B, num_categories)
        """
        fused = self.fusion(torch.cat([latent, caption_embed], dim=-1))
        logits = self.classifier(fused)
        return fused, logits


class PolicyCaptionLoss(nn.Module):
    """
    Combined loss for policy-aware caption generation.

    Equation (paper §3.2):
        L = λ_ce * L_CE(category) + λ_align * L_align(caption, gt_caption)

    where L_align = cosine similarity loss between generated and GT caption embeddings.
    """

    def __init__(self, lambda_ce: float = 1.0, lambda_align: float = 0.5):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_align = lambda_align

    def forward(
        self,
        category_logits: torch.Tensor,
        labels: torch.Tensor,
        caption_embed: torch.Tensor,
        gt_caption_embed: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(category_logits, labels)
        # Cosine alignment: maximize similarity to GT caption embedding
        cos_sim = F.cosine_similarity(caption_embed, gt_caption_embed, dim=-1)
        align_loss = (1 - cos_sim).mean()
        return self.lambda_ce * ce_loss + self.lambda_align * align_loss
