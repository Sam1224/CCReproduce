"""
UNIVID toy model.

Architecture (faithful to UNIVID §3 at toy scale):

  1. Visual Encoder        : frame-level MLP → mean-pool across frames
  2. Policy Embedding      : learnable policy-type embedding
  3. Caption Decoder       : small transformer decoder for policy-aware caption generation
  4. Violation Head        : binary MLP applied to the caption representation

Real UNIVID uses a large-scale VLM (InternVL / LLaVA class) fine-tuned at ByteDance.
This toy substitutes the heavy backbone with trainable linear layers while preserving
the key interfaces:
  - policy-conditioned generation (policy prefix injected into decoder context)
  - caption-based violation signal (violation head reads from caption pooled state)
  - mixed label weighting in loss computation

Equations:
  v = MeanPool(MLP_vis(F))           # (B, D_h)  visual context    [§3.1]
  p = E_policy(policy_id)             # (B, D_h)  policy context    [§3.1]
  ctx = LayerNorm(v + p)              # (B, D_h)  fused context     [§3.1]
  caption_logits = TransDec(ctx, cap) # (B, T, V) autoregressive    [§3.2]
  viol_logit = MLP_viol(cap_pooled)   # (B,)      violation head    [§3.4]
"""

import torch
import torch.nn as nn
from data import VOCAB_SIZE, POLICIES


class VisualEncoder(nn.Module):
    """Frame MLP → mean-pool."""

    def __init__(self, frame_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_frames, frame_dim) → (B, hidden_dim)
        return self.proj(x).mean(dim=1)


class CaptionDecoder(nn.Module):
    """
    Single-layer transformer decoder for caption generation.
    Context (visual + policy) is injected as a key-value source in cross-attention.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, n_heads: int = 4, max_seq: int = 20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        caption_ids: torch.Tensor,  # (B, T)
        context: torch.Tensor,      # (B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, T, vocab_size)  for next-token prediction
            cap_pooled: (B, hidden_dim) mean-pooled caption states for violation head
        """
        B, T = caption_ids.shape
        pos = torch.arange(T, device=caption_ids.device).unsqueeze(0)
        h = self.embed(caption_ids) + self.pos_embed(pos)

        # Self-attention (causal mask for AR generation)
        causal_mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        h2, _ = self.self_attn(h, h, h, attn_mask=causal_mask)
        h = self.norm1(h + h2)

        # Cross-attention: context as key/value
        ctx_kv = context.unsqueeze(1)  # (B, 1, D)
        h2, _ = self.cross_attn(h, ctx_kv, ctx_kv)
        h = self.norm2(h + h2)

        h = self.norm3(h + self.ff(h))
        logits = self.out_proj(h)              # (B, T, V)
        cap_pooled = h.mean(dim=1)             # (B, D)
        return logits, cap_pooled


class ViolationHead(nn.Module):
    """Binary violation classifier applied on pooled caption representation. [§3.4]"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cap_pooled: torch.Tensor) -> torch.Tensor:
        # (B, D) → (B,)
        return self.mlp(cap_pooled).squeeze(-1)


class UNIVID(nn.Module):
    """
    UNIVID: Unified Vision-Language Model for Video Moderation (toy).

    Policy-aware caption generation + violation detection from caption.
    """

    def __init__(
        self,
        frame_dim: int = 64,
        hidden_dim: int = 128,
        num_policies: int = len(POLICIES),
        vocab_size: int = VOCAB_SIZE,
        n_heads: int = 4,
        max_seq: int = 20,
    ):
        super().__init__()
        self.visual_enc = VisualEncoder(frame_dim, hidden_dim)
        self.policy_emb = nn.Embedding(num_policies, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.caption_dec = CaptionDecoder(vocab_size, hidden_dim, n_heads, max_seq)
        self.violation_head = ViolationHead(hidden_dim)

    def forward(
        self,
        video_feat: torch.Tensor,   # (B, T_frames, frame_dim)
        policy_id: torch.Tensor,    # (B,)
        caption_ids: torch.Tensor,  # (B, T_seq)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with:
          caption_logits: (B, T_seq, vocab_size)
          viol_logit:     (B,)
          cap_pooled:     (B, hidden_dim)
        """
        # §3.1: Fuse visual + policy context
        v = self.visual_enc(video_feat)           # (B, D)
        p = self.policy_emb(policy_id)            # (B, D)
        ctx = self.context_norm(v + p)            # (B, D)

        # §3.2: Policy-conditioned caption generation
        caption_logits, cap_pooled = self.caption_dec(caption_ids, ctx)

        # §3.4: Violation detection from caption representation
        viol_logit = self.violation_head(cap_pooled)

        return {
            "caption_logits": caption_logits,
            "viol_logit": viol_logit,
            "cap_pooled": cap_pooled,
        }

    @torch.no_grad()
    def predict_violation(
        self,
        video_feat: torch.Tensor,
        policy_id: torch.Tensor,
        caption_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Returns binary violation predictions (B,)."""
        out = self.forward(video_feat, policy_id, caption_ids)
        return (out["viol_logit"] > 0.0).long()
