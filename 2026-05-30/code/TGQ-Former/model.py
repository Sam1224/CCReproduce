"""
Reproduction: TGQ-Former — Text-Guided Visual Representation Learning
for Robust Multimodal E-Commerce Recommendation
arXiv: 2605.17366

Architecture:
  - Text-Guided Q-Former with dual-stream visual disentanglement
  - Metadata-Anchored Semantic Queries (capture metadata-consistent evidence)
  - Exploratory Queries (capture complementary visual patterns)
  - Dual-Gated Vector Modulation (DGVM)
  - Redundancy-Reduction Regularizer (R³)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ---------------------------------------------------------------------------
# 1. Text Encoder (metadata encoder)
# ---------------------------------------------------------------------------

class MetadataTextEncoder(nn.Module):
    """
    Encodes product metadata (title, category, attributes) into a text representation.
    In practice: uses a pre-trained BERT or domain-specific text encoder.
    """

    def __init__(self, text_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        # Pseudocode: self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        # For reproduction: linear projection from text feature space
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat: (batch, text_dim) — pre-computed metadata embedding
        Returns:
            metadata_repr: (batch, hidden_dim)
        """
        return self.proj(text_feat)


# ---------------------------------------------------------------------------
# 2. Visual Encoder (frozen backbone)
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """
    Pre-trained visual backbone (ViT or CLIP) — frozen during TGQ-Former training.
    Provides visual feature tokens for the Q-Former to attend to.
    """

    def __init__(self, visual_dim: int = 1024, num_visual_tokens: int = 196):
        super().__init__()
        # In practice: use frozen ViT or CLIP visual encoder
        # For reproduction: simulate with a linear projection
        self.visual_dim = visual_dim
        self.num_visual_tokens = num_visual_tokens
        self.patch_proj = nn.Linear(visual_dim, visual_dim)

    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat: (batch, num_visual_tokens, visual_dim) — visual patch features
        Returns:
            visual_tokens: (batch, num_visual_tokens, visual_dim)
        """
        return self.patch_proj(visual_feat)


# ---------------------------------------------------------------------------
# 3. Text-Guided Q-Former (TGQ-Former) — Core Module
# ---------------------------------------------------------------------------

class TGQFormerLayer(nn.Module):
    """
    A single TGQ-Former layer implementing:
        1. Self-attention among query tokens
        2. Cross-attention from queries to visual tokens
        3. Text-guided conditioning via metadata representation
        4. FFN
    """

    def __init__(self, query_dim: int = 256, visual_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        # Cross-attention: queries attend to visual tokens
        self.cross_attn = nn.MultiheadAttention(
            query_dim, num_heads, batch_first=True,
            kdim=visual_dim, vdim=visual_dim,
        )
        # Text conditioning: metadata biases the query representations
        self.text_condition_proj = nn.Linear(256, query_dim)  # metadata_dim → query_dim
        self.text_gate = nn.Linear(query_dim, query_dim)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,         # (batch, num_queries, query_dim)
        visual_tokens: torch.Tensor,   # (batch, num_tokens, visual_dim)
        metadata_repr: torch.Tensor,   # (batch, query_dim)
    ) -> torch.Tensor:
        """Returns updated query representations: (batch, num_queries, query_dim)"""
        # 1. Self-attention
        q = self.norm1(queries)
        q_attn, _ = self.self_attn(q, q, q)
        queries = queries + q_attn

        # 2. Text-guided conditioning (metadata modulates the queries)
        # This is the key innovation: metadata guides what visual features to extract
        metadata_bias = self.text_condition_proj(metadata_repr)  # (batch, query_dim)
        gate = torch.sigmoid(self.text_gate(metadata_bias)).unsqueeze(1)  # (batch, 1, query_dim)
        queries = queries * (1 + gate)

        # 3. Cross-attention: queries attend to visual tokens
        q2 = self.norm2(queries)
        q_cross, _ = self.cross_attn(q2, visual_tokens, visual_tokens)
        queries = queries + q_cross

        # 4. FFN
        q3 = self.norm3(queries)
        queries = queries + self.ffn(q3)

        return queries


class TGQFormerDualStream(nn.Module):
    """
    Dual-stream TGQ-Former producing:
        Stream A: Metadata-Anchored Semantic Queries (Q_meta)
          - Strongly guided by metadata; extracts metadata-consistent visual evidence
        Stream B: Exploratory Queries (Q_explore)
          - Weakly guided; captures visual patterns beyond metadata description

    Equations (Section 3.2):
        Q_meta = TGQFormer(Q_init_A, V, metadata; strong_guidance=True)
        Q_explore = TGQFormer(Q_init_B, V, metadata; strong_guidance=False)
    """

    def __init__(
        self,
        num_meta_queries: int = 16,
        num_explore_queries: int = 16,
        query_dim: int = 256,
        visual_dim: int = 1024,
        metadata_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        # Learnable query tokens for both streams
        self.meta_queries = nn.Parameter(torch.randn(1, num_meta_queries, query_dim) * 0.02)
        self.explore_queries = nn.Parameter(torch.randn(1, num_explore_queries, query_dim) * 0.02)

        # Separate Q-Former layers for each stream
        self.meta_layers = nn.ModuleList([
            TGQFormerLayer(query_dim, visual_dim, num_heads) for _ in range(num_layers)
        ])
        self.explore_layers = nn.ModuleList([
            TGQFormerLayer(query_dim, visual_dim, num_heads) for _ in range(num_layers)
        ])

        # Stronger/weaker text conditioning scale
        self.meta_guidance_scale = nn.Parameter(torch.tensor(1.5))   # stronger guidance
        self.explore_guidance_scale = nn.Parameter(torch.tensor(0.5)) # weaker guidance

    def forward(
        self,
        visual_tokens: torch.Tensor,  # (batch, num_tokens, visual_dim)
        metadata_repr: torch.Tensor,  # (batch, metadata_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_meta: (batch, num_meta_queries, query_dim)
            q_explore: (batch, num_explore_queries, query_dim)
        """
        batch_size = visual_tokens.shape[0]

        # Initialize queries
        q_meta = self.meta_queries.expand(batch_size, -1, -1)
        q_explore = self.explore_queries.expand(batch_size, -1, -1)

        # Scale metadata guidance differently for each stream
        meta_cond = metadata_repr * self.meta_guidance_scale
        explore_cond = metadata_repr * self.explore_guidance_scale

        # Forward through respective layers
        for layer in self.meta_layers:
            q_meta = layer(q_meta, visual_tokens, meta_cond)

        for layer in self.explore_layers:
            q_explore = layer(q_explore, visual_tokens, explore_cond)

        return q_meta, q_explore


# ---------------------------------------------------------------------------
# 4. Dual-Gated Vector Modulation (DGVM)
# ---------------------------------------------------------------------------

class DualGatedVectorModulation(nn.Module):
    """
    DGVM: Calibrates the two query streams using:
        Gate_1: Cross-modal agreement (text-visual alignment score)
        Gate_2: Image-derived visual saliency cues

    Formula (Eq. 6 in paper):
        agreement_score = cosine_sim(metadata_repr, pool(Q_meta))
        saliency = VisualSaliencyNet(visual_tokens)
        gate_meta = sigmoid(W_meta * [agreement_score; saliency])
        gate_explore = sigmoid(W_explore * [1 - agreement_score; saliency])
        Q_meta_calibrated = gate_meta * Q_meta
        Q_explore_calibrated = gate_explore * Q_explore
    """

    def __init__(self, query_dim: int = 256, metadata_dim: int = 256, visual_dim: int = 1024):
        super().__init__()
        # Visual saliency network
        self.saliency_net = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Gate networks
        # Input: [agreement_score (1), saliency (1)]
        self.meta_gate = nn.Sequential(
            nn.Linear(2, query_dim),
            nn.Sigmoid(),
        )
        self.explore_gate = nn.Sequential(
            nn.Linear(2, query_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        q_meta: torch.Tensor,         # (batch, num_meta_q, query_dim)
        q_explore: torch.Tensor,      # (batch, num_explore_q, query_dim)
        metadata_repr: torch.Tensor,  # (batch, metadata_dim)
        visual_tokens: torch.Tensor,  # (batch, num_tokens, visual_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns calibrated query streams: (q_meta_cal, q_explore_cal)
        """
        # 1. Cross-modal agreement score
        q_meta_pooled = q_meta.mean(dim=1)  # (batch, query_dim)
        # Normalize for cosine similarity
        meta_norm = F.normalize(metadata_repr, dim=-1)
        q_norm = F.normalize(q_meta_pooled, dim=-1)
        agreement = (meta_norm * q_norm).sum(dim=-1, keepdim=True)  # (batch, 1)

        # 2. Visual saliency (max-pooled over tokens)
        token_saliency = self.saliency_net(visual_tokens)  # (batch, num_tokens, 1)
        saliency = token_saliency.max(dim=1).values        # (batch, 1)

        # 3. Gate computation
        gate_input = torch.cat([agreement, saliency], dim=-1)  # (batch, 2)
        anti_gate_input = torch.cat([1 - agreement, saliency], dim=-1)

        gate_meta = self.meta_gate(gate_input).unsqueeze(1)       # (batch, 1, query_dim)
        gate_explore = self.explore_gate(anti_gate_input).unsqueeze(1)  # (batch, 1, query_dim)

        return q_meta * gate_meta, q_explore * gate_explore


# ---------------------------------------------------------------------------
# 5. Redundancy-Reduction Regularizer (R³)
# ---------------------------------------------------------------------------

class RedundancyReductionRegularizer(nn.Module):
    """
    R³ Loss: Encourages complementarity between the two query streams.
    Penalizes high correlation between Q_meta and Q_explore representations.

    Formula (Eq. 8 in paper):
        C = cosine_sim_matrix(pool(Q_meta), pool(Q_explore))  [per batch]
        L_R3 = mean(off_diagonal(C²))
    """

    def forward(
        self,
        q_meta: torch.Tensor,    # (batch, num_q, dim)
        q_explore: torch.Tensor, # (batch, num_q, dim)
    ) -> torch.Tensor:
        """Returns redundancy penalty scalar."""
        # Pool to (batch, dim)
        p_meta = F.normalize(q_meta.mean(dim=1), dim=-1)
        p_explore = F.normalize(q_explore.mean(dim=1), dim=-1)

        # Cross-correlation matrix: (batch, dim, dim)
        # Off-diagonal elements should be near zero for complementarity
        cross_corr = torch.bmm(p_meta.unsqueeze(-1), p_explore.unsqueeze(-2))  # (batch, dim, dim)

        # Off-diagonal penalty
        batch_size, dim, _ = cross_corr.shape
        mask = ~torch.eye(dim, dtype=torch.bool, device=cross_corr.device).unsqueeze(0)
        off_diag = cross_corr[mask.expand_as(cross_corr)]
        return off_diag.pow(2).mean()


# ---------------------------------------------------------------------------
# 6. Full TGQ-Former Model for E-Commerce I2I Retrieval
# ---------------------------------------------------------------------------

class TGQFormerRetrieval(nn.Module):
    """
    Full TGQ-Former model for e-commerce item-to-item retrieval.

    Pipeline:
        1. Encode metadata → metadata_repr
        2. Encode visual → visual_tokens (frozen backbone)
        3. TGQ-Former dual-stream → Q_meta, Q_explore
        4. DGVM calibration → Q_meta_cal, Q_explore_cal
        5. Pool and concatenate → item_embedding
        6. Contrastive loss for retrieval training
    """

    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 1024,
        num_visual_tokens: int = 196,
        query_dim: int = 256,
        num_meta_queries: int = 16,
        num_explore_queries: int = 16,
        num_layers: int = 6,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.metadata_encoder = MetadataTextEncoder(text_dim, query_dim)
        self.visual_encoder = VisualEncoder(visual_dim, num_visual_tokens)

        self.dual_stream = TGQFormerDualStream(
            num_meta_queries=num_meta_queries,
            num_explore_queries=num_explore_queries,
            query_dim=query_dim,
            visual_dim=visual_dim,
            metadata_dim=query_dim,
            num_layers=num_layers,
        )

        self.dgvm = DualGatedVectorModulation(query_dim, query_dim, visual_dim)
        self.r3_loss = RedundancyReductionRegularizer()

        # Final projection to embedding space
        fused_dim = query_dim * 2  # [Q_meta_pooled; Q_explore_pooled]
        self.item_proj = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def encode_item(
        self,
        text_feat: torch.Tensor,    # (batch, text_dim)
        visual_feat: torch.Tensor,  # (batch, num_tokens, visual_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: (batch, embedding_dim) — item embedding for retrieval
            r3_penalty: scalar — redundancy reduction loss for training
        """
        # 1. Encode metadata and visual
        metadata_repr = self.metadata_encoder(text_feat)
        visual_tokens = self.visual_encoder(visual_feat)

        # 2. Dual-stream Q-Former
        q_meta, q_explore = self.dual_stream(visual_tokens, metadata_repr)

        # 3. DGVM calibration
        q_meta_cal, q_explore_cal = self.dgvm(q_meta, q_explore, metadata_repr, visual_tokens)

        # 4. Pool each stream
        p_meta = q_meta_cal.mean(dim=1)       # (batch, query_dim)
        p_explore = q_explore_cal.mean(dim=1) # (batch, query_dim)

        # 5. Concatenate and project
        fused = torch.cat([p_meta, p_explore], dim=-1)  # (batch, 2*query_dim)
        embedding = self.item_proj(fused)               # (batch, embedding_dim)
        embedding = F.normalize(embedding, dim=-1)

        # 6. Redundancy penalty
        r3_penalty = self.r3_loss(q_meta_cal, q_explore_cal)

        return embedding, r3_penalty

    def forward(
        self,
        query_text: torch.Tensor,    # (batch, text_dim)
        query_visual: torch.Tensor,  # (batch, num_tokens, visual_dim)
        pos_text: torch.Tensor,      # (batch, text_dim)
        pos_visual: torch.Tensor,    # (batch, num_tokens, visual_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive I2I retrieval training.
        Returns: (contrastive_loss, r3_loss)
        """
        # Encode query and positive items
        q_emb, q_r3 = self.encode_item(query_text, query_visual)
        p_emb, p_r3 = self.encode_item(pos_text, pos_visual)

        # Contrastive loss (InfoNCE / NT-Xent)
        sim_matrix = torch.matmul(q_emb, p_emb.T) / 0.07  # (batch, batch), temp=0.07
        labels = torch.arange(q_emb.shape[0], device=q_emb.device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels)

        r3_penalty = (q_r3 + p_r3) / 2.0

        return contrastive_loss, r3_penalty


# ---------------------------------------------------------------------------
# 7. Full Training Loss
# ---------------------------------------------------------------------------

class TGQFormerLoss(nn.Module):
    """
    Total loss:
        L = L_contrastive + alpha * L_R3

    Where:
        L_contrastive: InfoNCE on item embeddings
        L_R3: Redundancy-reduction regularizer (encourages stream complementarity)
    """

    def __init__(self, alpha_r3: float = 0.1):
        super().__init__()
        self.alpha_r3 = alpha_r3

    def forward(
        self,
        contrastive_loss: torch.Tensor,
        r3_penalty: torch.Tensor,
    ) -> dict:
        total = contrastive_loss + self.alpha_r3 * r3_penalty
        return {
            "total": total,
            "contrastive": contrastive_loss,
            "r3": r3_penalty,
        }
