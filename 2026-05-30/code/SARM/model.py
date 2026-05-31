"""
Reproduction: SARM — LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking
arXiv: 2602.09401  |  Kuaishou Technology & Chinese Academy of Sciences

Architecture:
  Semantic Anchors = learnable text tokens jointly optimized with ranking
  LLM generates initial anchor content offline
  End-to-end ranking integrates anchors with behavioral/contextual features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. Semantic Anchor Representation
# ---------------------------------------------------------------------------

class SemanticAnchor(nn.Module):
    """
    Learnable semantic anchor for a live-stream session.

    A semantic anchor is represented as K learnable text token embeddings
    jointly trained with the ranking objective, initialized from an LLM-generated
    natural language description of the live-stream content.

    Per Section 3.1 of SARM paper:
        anchor = Embedding(token_1, ..., token_K)
        anchor_repr = Attention_Pool(anchor_tokens)
    """

    def __init__(
        self,
        num_anchor_tokens: int = 8,
        token_dim: int = 256,
        anchor_hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_anchor_tokens = num_anchor_tokens
        self.token_dim = token_dim

        # Learnable anchor token embeddings (K tokens per anchor)
        self.anchor_tokens = nn.Parameter(
            torch.randn(num_anchor_tokens, token_dim) * 0.02
        )

        # Self-attention pooling to get a single anchor representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, batch_first=True
        )
        self.pooling_proj = nn.Linear(token_dim, anchor_hidden_dim)

    def initialize_from_llm_embedding(self, llm_embedding: torch.Tensor):
        """
        Initialize anchor tokens from LLM-generated text embedding.
        llm_embedding: (token_dim,) or (num_anchor_tokens, token_dim)
        """
        with torch.no_grad():
            if llm_embedding.dim() == 1:
                # Broadcast to all anchor tokens with small random perturbation
                self.anchor_tokens.data = llm_embedding.unsqueeze(0).expand(
                    self.num_anchor_tokens, -1
                ) + torch.randn_like(self.anchor_tokens) * 0.01
            else:
                self.anchor_tokens.data = llm_embedding[: self.num_anchor_tokens]

    def forward(self) -> torch.Tensor:
        """
        Returns anchor representation: (anchor_hidden_dim,)
        """
        # Self-attention over anchor tokens: (1, K, D)
        tokens = self.anchor_tokens.unsqueeze(0)
        attended, _ = self.self_attention(tokens, tokens, tokens)
        # Mean pool: (1, D)
        pooled = attended.mean(dim=1)
        # Project: (anchor_hidden_dim,)
        return self.pooling_proj(pooled.squeeze(0))


# ---------------------------------------------------------------------------
# 2. LLM-based Anchor Content Generator (offline)
# ---------------------------------------------------------------------------

class LLMSemanticAnchorGenerator:
    """
    Offline component: Uses LLM to generate natural language descriptions
    of live-stream content segments, which initialize the semantic anchors.

    Granularity options (ablated in paper):
        - session: one anchor per live-stream session
        - topic: one anchor per content topic shift (best in ablation)
        - product: one anchor per featured product
    """

    ANCHOR_GENERATION_PROMPT = (
        "You are analyzing a live-streaming e-commerce session. "
        "Given the following transcript segment and visual description, "
        "generate a concise semantic description (2-3 sentences) that captures:\n"
        "1. The host's style and tone\n"
        "2. The products being featured\n"
        "3. The key selling points mentioned\n\n"
        "Transcript: {transcript}\n"
        "Visual description: {visual_desc}\n\n"
        "Semantic description:"
    )

    def __init__(self, llm_api_or_model=None):
        self.llm = llm_api_or_model

    def generate_anchor_description(
        self,
        transcript: str,
        visual_desc: str = "",
    ) -> str:
        """Generate semantic anchor description for a content segment."""
        prompt = self.ANCHOR_GENERATION_PROMPT.format(
            transcript=transcript[:1000],  # truncate for token limits
            visual_desc=visual_desc[:200],
        )
        if self.llm is not None:
            # Call actual LLM
            # return self.llm.generate(prompt)
            pass

        # Placeholder for demonstration
        return (
            f"The host presents various products with an energetic, promotional style. "
            f"Featured items include: {transcript[:50]}... "
            f"Key selling points: quality guarantee, limited-time pricing, and free shipping."
        )

    def embed_description(self, description: str, text_encoder) -> torch.Tensor:
        """
        Encode anchor description text to embedding for anchor initialization.
        Returns: (token_dim,)
        """
        # Pseudocode:
        # tokens = text_encoder.tokenize(description)
        # with torch.no_grad():
        #     embedding = text_encoder(tokens).last_hidden_state[:, 0, :]
        # return embedding.squeeze(0)
        return torch.randn(256)  # placeholder (token_dim=256)


# ---------------------------------------------------------------------------
# 3. Live-Stream Content Encoder
# ---------------------------------------------------------------------------

class LiveStreamContentEncoder(nn.Module):
    """
    Encodes real-time live-stream signals:
        - Speech/transcript (text)
        - Visual frames (images)
        - User interaction signals (comments, likes)
    """

    def __init__(self, content_dim: int = 256):
        super().__init__()
        # Text encoder for transcript
        self.text_proj = nn.Sequential(
            nn.Linear(768, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
        )
        # Visual encoder placeholder
        self.visual_proj = nn.Sequential(
            nn.Linear(2048, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
        )
        # Interaction encoder
        self.interaction_proj = nn.Sequential(
            nn.Linear(64, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
        )
        # Multimodal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=content_dim, num_heads=4, batch_first=True
        )

    def forward(
        self,
        text_feat: torch.Tensor,       # (batch, 768) — transcript BERT embedding
        visual_feat: torch.Tensor,      # (batch, 2048) — frame ResNet/CLIP embedding
        interaction_feat: torch.Tensor, # (batch, 64) — like/comment counts etc.
    ) -> torch.Tensor:
        """Returns fused content representation: (batch, content_dim)"""
        t = self.text_proj(text_feat).unsqueeze(1)       # (batch, 1, D)
        v = self.visual_proj(visual_feat).unsqueeze(1)   # (batch, 1, D)
        i = self.interaction_proj(interaction_feat).unsqueeze(1)  # (batch, 1, D)

        # Concatenate modalities: (batch, 3, D)
        multi_modal = torch.cat([t, v, i], dim=1)
        # Cross-modal attention
        attended, _ = self.fusion(multi_modal, multi_modal, multi_modal)
        return attended.mean(dim=1)  # (batch, D)


# ---------------------------------------------------------------------------
# 4. SARM Ranking Model
# ---------------------------------------------------------------------------

class SARMRankingModel(nn.Module):
    """
    SARM: End-to-end live-streaming ranking model with semantic anchors.

    Ranking Score Computation (Eq. 3 in paper):
        h_content = ContentEncoder(transcript, visual, interaction)
        h_anchor = SemanticAnchor()  [per stream, shared across users]
        h_user = UserEncoder(user_features)
        h_context = ContextEncoder(context_features)
        score = RankingHead(concat(h_content, h_anchor, h_user, h_context))
    """

    def __init__(
        self,
        num_anchor_tokens: int = 8,
        content_dim: int = 256,
        user_dim: int = 128,
        context_dim: int = 64,
        anchor_hidden_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.content_encoder = LiveStreamContentEncoder(content_dim)
        self.semantic_anchor = SemanticAnchor(num_anchor_tokens, content_dim, anchor_hidden_dim)

        # User feature encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Context feature encoder (time of day, device, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Anchor-content interaction
        self.anchor_content_interaction = nn.MultiheadAttention(
            embed_dim=content_dim, num_heads=4, batch_first=True
        )

        # Final ranking head
        combined_dim = content_dim + anchor_hidden_dim + hidden_dim + hidden_dim // 2
        self.ranking_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        text_feat: torch.Tensor,        # (batch, 768)
        visual_feat: torch.Tensor,      # (batch, 2048)
        interaction_feat: torch.Tensor, # (batch, 64)
        user_feat: torch.Tensor,        # (batch, 128)
        context_feat: torch.Tensor,     # (batch, 64)
    ) -> torch.Tensor:
        """
        Returns:
            ranking_score: (batch,) — predicted relevance/engagement score
        """
        # 1. Encode live-stream content
        h_content = self.content_encoder(text_feat, visual_feat, interaction_feat)  # (batch, D)

        # 2. Get semantic anchor representation (shared across all users for this stream)
        h_anchor = self.semantic_anchor()  # (anchor_hidden_dim,)
        h_anchor_expanded = h_anchor.unsqueeze(0).expand(h_content.shape[0], -1)  # (batch, D_a)

        # 3. Anchor-content cross-attention (anchor conditions on content)
        content_for_attn = h_content.unsqueeze(1)                    # (batch, 1, D)
        anchor_for_attn = h_anchor.unsqueeze(0).unsqueeze(0).expand(
            h_content.shape[0], 1, -1
        )                                                             # (batch, 1, D)
        # NOTE: anchor dim must match content_dim for attention; proj if needed
        # Here we skip the cross-attention for simplicity and use concatenation

        # 4. User and context encoding
        h_user = self.user_encoder(user_feat)       # (batch, hidden_dim)
        h_context = self.context_encoder(context_feat)  # (batch, hidden_dim//2)

        # 5. Concatenate all representations
        combined = torch.cat([h_content, h_anchor_expanded, h_user, h_context], dim=-1)

        # 6. Final ranking score
        return self.ranking_head(combined).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# 5. SARM Training Objective
# ---------------------------------------------------------------------------

class SARMTrainingLoss(nn.Module):
    """
    Joint ranking loss optimized over content encoding + semantic anchors.
    Uses pairwise ranking loss (BPR) + point-wise BCE.

    The semantic anchor parameters are part of the computation graph,
    so gradients flow back through the anchor during ranking optimization.
    """

    def __init__(self, bpr_weight: float = 0.5):
        super().__init__()
        self.bpr_weight = bpr_weight
        self.bce = nn.BCEWithLogitsLoss()

    def bpr_loss(
        self,
        pos_scores: torch.Tensor,  # (batch,)
        neg_scores: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """Bayesian Personalized Ranking loss for positive/negative pairs."""
        # L_BPR = -log(sigmoid(pos_score - neg_score))
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        pos_labels: torch.Tensor,
        neg_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # BPR pairwise ranking
        l_bpr = self.bpr_loss(pos_scores, neg_scores)

        # Point-wise BCE on both positive and negative examples
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        l_bce = self.bce(all_scores, all_labels.float())

        total = l_bce + self.bpr_weight * l_bpr
        return {"total": total, "bpr": l_bpr, "bce": l_bce}
