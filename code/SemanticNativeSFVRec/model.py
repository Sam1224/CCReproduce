"""
SemanticNativeSFVRec: Semantic-Native Long-Sequence Short-Form Video Recommendation
Paper: Beyond Item IDs — Semantic-Native Long-Sequence Modeling for Billion-User
       Short-Form Video Recommendation (arXiv 2606.07546, Google)

Key ideas:
  1. Residual-Quantization (RQ-VAE) Semantic IDs replace atomic Video IDs
  2. Depth-truncated coarse-grained SIDs compress embedding tables from billions → millions
  3. Shared semantic prefix enables zero-shot cold-start generalization
  4. Long-sequence Transformer user model over SID sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SemanticSFVConfig:
    # Semantic ID codebook config
    content_dim: int = 256          # dimension of content embedding (e.g., from CLIP)
    num_rq_levels: int = 4          # total RQ-VAE levels
    truncation_depth: int = 2       # keep only first D levels (depth-truncation)
    codebook_size: int = 2048       # codes per level

    # User sequence model
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 200          # max history length
    dropout: float = 0.1

    # Prediction head
    num_items: int = 100000         # catalog size (for in-batch softmax)


class VectorQuantizer(nn.Module):
    """Single-level VQ codebook used in RQ-VAE."""

    def __init__(self, num_codes: int, dim: int):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.normal_(self.codebook.weight, std=0.02)
        self.num_codes = num_codes

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [..., dim]
        Returns: (quantized [..., dim], indices [...])
        """
        flat = z.reshape(-1, z.size(-1))
        # Compute distances
        dist = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2 * flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(-1)
        )
        idx = dist.argmin(-1)  # [N]
        quantized = self.codebook(idx).reshape(z.shape)
        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        return quantized_st, idx.reshape(z.shape[:-1])

    def commitment_loss(self, z: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z, quantized.detach()) + 0.25 * F.mse_loss(quantized, z.detach())


class RQVAEEncoder(nn.Module):
    """
    Residual Quantization VAE for generating hierarchical Semantic IDs from content embeddings.

    RQ process (L levels):
      r_0 = content_emb
      idx_l, q_l = VQ(r_{l-1})
      r_l = r_{l-1} - q_l    ← residual for next level

    Depth-truncated SID = (idx_0, ..., idx_{D-1})  where D <= L
    """

    def __init__(self, config: SemanticSFVConfig):
        super().__init__()
        self.num_levels = config.num_rq_levels
        self.truncation_depth = config.truncation_depth
        self.codebook_size = config.codebook_size

        self.input_proj = nn.Linear(config.content_dim, config.hidden_size)
        self.vq_levels = nn.ModuleList([
            VectorQuantizer(config.codebook_size, config.hidden_size)
            for _ in range(config.num_rq_levels)
        ])

    def forward(
        self, content_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        content_emb: [B, content_dim]
        Returns:
          semantic_ids: [B, truncation_depth] — truncated SIDs
          all_ids:      [B, num_levels]       — full SIDs (for training)
          vq_loss:      scalar
        """
        z = self.input_proj(content_emb)  # [B, H]
        residual = z
        all_ids = []
        vq_loss = torch.tensor(0.0, device=z.device)

        for vq in self.vq_levels:
            q, idx = vq(residual)
            vq_loss = vq_loss + vq.commitment_loss(residual, q)
            all_ids.append(idx)
            residual = residual - q.detach()

        all_ids = torch.stack(all_ids, dim=-1)  # [B, L]
        semantic_ids = all_ids[:, :self.truncation_depth]  # [B, D]
        return semantic_ids, all_ids, vq_loss


class SemanticIDEmbedding(nn.Module):
    """
    Depth-truncated Semantic ID embedding.

    Each SID = (idx_0, idx_1, ..., idx_{D-1}).
    Embedding = sum of per-level embeddings (with level-specific offsets to avoid collision).

    Table size: D * codebook_size  instead of  N (billions of atomic IDs).
    """

    def __init__(self, config: SemanticSFVConfig):
        super().__init__()
        self.depth = config.truncation_depth
        self.codebook_size = config.codebook_size
        # One embedding table per level
        self.level_embs = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.hidden_size)
            for _ in range(config.truncation_depth)
        ])
        self.level_weights = nn.Parameter(torch.ones(config.truncation_depth))

    def forward(self, semantic_ids: torch.Tensor) -> torch.Tensor:
        """
        semantic_ids: [B, L, D] or [B, D]
        Returns: [B, L, H] or [B, H] item embeddings
        """
        original_shape = semantic_ids.shape[:-1]
        ids = semantic_ids.reshape(-1, self.depth)  # [N, D]

        weights = F.softmax(self.level_weights, dim=0)
        emb = sum(
            weights[d] * self.level_embs[d](ids[:, d])
            for d in range(self.depth)
        )  # [N, H]
        return emb.reshape(*original_shape, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, config: SemanticSFVConfig):
        super().__init__()
        self.emb = nn.Embedding(config.max_seq_len, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        return x + self.emb(pos)


class LongSeqTransformer(nn.Module):
    """
    Causal Transformer for long-sequence user history modeling over Semantic IDs.
    """

    def __init__(self, config: SemanticSFVConfig):
        super().__init__()
        self.item_emb = SemanticIDEmbedding(config)
        self.pos_enc = PositionalEncoding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        semantic_ids: torch.Tensor,   # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] float mask (1=valid)
    ) -> torch.Tensor:
        """Returns [B, H] user representation (last valid position)."""
        x = self.item_emb(semantic_ids)  # [B, L, H]
        x = self.pos_enc(x)
        x = self.dropout(self.layer_norm(x))

        # Causal mask
        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.output_norm(x)

        # Extract last valid position as user representation
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=-1).long() - 1  # [B]
            lengths = lengths.clamp(min=0)
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.size(-1))
            user_repr = x.gather(1, idx).squeeze(1)  # [B, H]
        else:
            user_repr = x[:, -1, :]

        return user_repr  # [B, H]


class SemanticNativeSFVRec(nn.Module):
    """
    Full SemanticNativeSFVRec model.

    Training:
      1. Content encoder (frozen) → content_emb [B, content_dim]
      2. RQ-VAE → semantic_ids [B, D]  (depth-truncated)
      3. History of semantic_ids → LongSeqTransformer → user_repr [B, H]
      4. In-batch softmax loss vs. target item semantic_emb [B, H]

    Cold-start:
      New items have no atomic ID but can be encoded by content encoder → RQ-VAE → SID
      Shared codebook embeddings transfer immediately.
    """

    def __init__(self, config: SemanticSFVConfig):
        super().__init__()
        self.config = config
        self.rqvae = RQVAEEncoder(config)
        self.seq_model = LongSeqTransformer(config)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_items(
        self, content_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """content_emb: [B, content_dim] → semantic_ids [B, D], full_ids [B, L], vq_loss"""
        return self.rqvae(content_emb)

    def forward(
        self,
        history_content_embs: torch.Tensor,  # [B, L, content_dim] history item content embeddings
        attention_mask: torch.Tensor,          # [B, L]
        target_content_emb: torch.Tensor,      # [B, content_dim] target item
    ) -> dict:
        """
        Full training forward pass.
        Returns loss dict with vq_loss, contrastive_loss, total_loss.
        """
        B, L, C = history_content_embs.shape

        # Step 1: Encode history items → semantic IDs
        flat_content = history_content_embs.reshape(B * L, C)
        hist_sids, _, hist_vq_loss = self.rqvae(flat_content)  # [B*L, D]
        hist_sids = hist_sids.reshape(B, L, -1)  # [B, L, D]

        # Step 2: Encode target item
        tgt_sids, _, tgt_vq_loss = self.rqvae(target_content_emb)  # [B, D]

        vq_loss = (hist_vq_loss + tgt_vq_loss) * 0.5

        # Step 3: User sequence modeling
        user_repr = self.seq_model(hist_sids, attention_mask)  # [B, H]

        # Step 4: Target item embedding
        tgt_emb = self.seq_model.item_emb(tgt_sids)  # [B, H]

        # Step 5: In-batch contrastive loss
        user_repr = F.normalize(user_repr, dim=-1)
        tgt_emb = F.normalize(tgt_emb, dim=-1)

        logits = user_repr @ tgt_emb.T / self.temperature.abs().clamp(min=1e-4)  # [B, B]
        labels = torch.arange(B, device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        total_loss = contrastive_loss + 0.1 * vq_loss

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "vq_loss": vq_loss,
        }

    @torch.no_grad()
    def get_user_repr(
        self,
        history_content_embs: torch.Tensor,  # [B, L, content_dim]
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: get user representation from history."""
        B, L, C = history_content_embs.shape
        flat_content = history_content_embs.reshape(B * L, C)
        hist_sids, _, _ = self.rqvae(flat_content)
        hist_sids = hist_sids.reshape(B, L, -1)
        return F.normalize(self.seq_model(hist_sids, attention_mask), dim=-1)

    @torch.no_grad()
    def get_item_emb(self, content_emb: torch.Tensor) -> torch.Tensor:
        """Inference: encode a batch of items → normalized item embeddings."""
        sids, _, _ = self.rqvae(content_emb)
        return F.normalize(self.seq_model.item_emb(sids), dim=-1)
