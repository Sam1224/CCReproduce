"""
FLUID Ranking Model — ID-free late-fusion design.

Replaces item ID embeddings with LUCID discrete codes as independent tokens.
The ranker ingests:
  - user embedding (from user ID)
  - room-level LUCID codes as independent tokens
  - slice-level LUCID codes as independent tokens (optional for toy)

Paper (Section 3.2):
    score = Ranker(e_user, [e_room_lucid_1, ..., e_room_lucid_K], context)
    where e_room_lucid_k = LUCIDCodeEmbed(c_k)  # no item ID used
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucid_encoder import LUCIDEncoder


class FLUIDRanker(nn.Module):
    """
    FLUID: ID-free Ranker using LUCID codes.

    Late-fusion design:
    1. LUCID encoder produces K discrete codes per room
    2. Each code is embedded via a per-level embedding table
    3. User embedding + LUCID code embeddings are fused for ranking score
    """

    def __init__(
        self,
        n_users: int,
        user_dim: int = 64,
        visual_dim: int = 256,
        audio_dim: int = 128,
        text_dim: int = 128,
        lucid_hidden_dim: int = 256,
        n_levels: int = 4,
        codebook_size: int = 512,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # User embedding table (standard)
        self.user_embed = nn.Embedding(n_users, user_dim)

        # LUCID encoder: produces hierarchical codes, no item ID
        self.lucid_encoder = LUCIDEncoder(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=lucid_hidden_dim,
            n_levels=n_levels,
            codebook_size=codebook_size,
        )

        # Per-level embedding tables for LUCID codes
        # Each code at level k is embedded independently (late-fusion design)
        self.code_embeds = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_dim) for _ in range(n_levels)]
        )
        self.n_levels = n_levels

        # Fusion and scoring MLP
        fused_dim = user_dim + n_levels * hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        visual: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
    ):
        """
        Args:
            user_ids: (B,) user indices
            visual:   (B, visual_dim)
            audio:    (B, audio_dim)
            text:     (B, text_dim)
        Returns:
            scores: (B,) ranking scores
            vq_loss: quantization loss for training
        """
        # User embedding
        e_user = self.user_embed(user_ids)                  # (B, user_dim)

        # LUCID codes — ID-free content encoding
        codes, vq_loss, _ = self.lucid_encoder(visual, audio, text)
        # codes: (B, K) discrete indices

        # Embed each level's code independently (late-fusion per paper)
        code_vecs = []
        for k, embed in enumerate(self.code_embeds):
            code_vecs.append(embed(codes[:, k]))            # (B, hidden_dim)
        code_vecs = torch.cat(code_vecs, dim=-1)            # (B, K*hidden_dim)

        # Fuse user + LUCID code embeddings
        fused = torch.cat([e_user, code_vecs], dim=-1)      # (B, user_dim + K*hidden_dim)
        scores = self.scorer(fused).squeeze(-1)             # (B,)

        return scores, vq_loss
