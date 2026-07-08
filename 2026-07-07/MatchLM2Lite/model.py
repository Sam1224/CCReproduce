"""
MatchLM2Lite — Model Architecture

Implements:
1. MatchLM: MLLM teacher for reproduced content identification
   - Three-modal joint encoding (video + audio + text)
   - Pairwise semantic representation extraction
   - Fine-grained reproduction scoring
2. MatchLite: Lightweight distilled student
   - Compact three-modal fusion
   - 35× cheaper inference than MatchLM
3. Distillation training objective

Key insight from paper:
"We utilize the MLLM as a paired video representation extractor rather than
as a next-token prediction generative model, which enables direct use of rich
semantic embeddings for discriminative classification."

Paper: "MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification"
arXiv: https://arxiv.org/abs/2606.14786
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Common Building Blocks ───────────────────────────────────────────────────

class MultiHeadAttentionPooling(nn.Module):
    """Attention pooling over a sequence of features → single vector."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → output: (B, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


# ─── MatchLM: MLLM Teacher ────────────────────────────────────────────────────

class VideoModalEncoder(nn.Module):
    """Visual feature encoder for MatchLM (processes frame-level features)."""
    def __init__(self, input_dim: int = 512, embed_dim: int = 512, num_layers: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim*4,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = MultiHeadAttentionPooling(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_frames, input_dim)
        x = self.proj(x)
        x = self.transformer(x)
        return self.pool(x)  # (B, embed_dim)


class AudioModalEncoder(nn.Module):
    """Audio feature encoder for MatchLM (processes segment-level features)."""
    def __init__(self, input_dim: int = 256, embed_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim*4,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = MultiHeadAttentionPooling(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_segments, input_dim)
        x = self.proj(x)
        x = self.transformer(x)
        return self.pool(x)


class TextModalEncoder(nn.Module):
    """Text encoder for MatchLM (processes title + description + captions)."""
    def __init__(self, vocab_size: int = 32000, embed_dim: int = 512,
                 max_len: int = 128, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim*4,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = MultiHeadAttentionPooling(embed_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(L, device=input_ids.device).unsqueeze(0))
        mask = attention_mask.eq(0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.pool(x)


class TrimodalFusion(nn.Module):
    """Fuse video + audio + text into a single pairwise embedding."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # Cross-modal attention: text attends to video + audio
        self.va_cross = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        visual: torch.Tensor,   # (B, D)
        audio: torch.Tensor,    # (B, D)
        text: torch.Tensor,     # (B, D)
    ) -> torch.Tensor:
        # Stack visual and audio for cross-attention
        va = torch.stack([visual, audio], dim=1)   # (B, 2, D)
        t = text.unsqueeze(1)                        # (B, 1, D)
        fused_t, _ = self.va_cross(t, va, va)
        fused_t = fused_t.squeeze(1)                 # (B, D)

        combined = torch.cat([visual, audio, fused_t], dim=-1)
        return self.norm(self.proj(combined))


class MatchLM(nn.Module):
    """
    MatchLM: MLLM-based teacher model for reproduced content identification.

    Design rationale:
    Instead of using the MLLM for generation, we use it as a *representation extractor*:
    each video is encoded into a rich semantic embedding via the three-modal encoders,
    and the reproduction score is computed from pairwise embedding similarity.

    This is the "teacher" model; MatchLite is trained to distill its embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        visual_input_dim: int = 512,
        audio_input_dim: int = 256,
        text_max_len: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Three modal encoders
        self.video_enc = VideoModalEncoder(visual_input_dim, embed_dim, num_layers=4)
        self.audio_enc = AudioModalEncoder(audio_input_dim, embed_dim, num_layers=2)
        self.text_enc = TextModalEncoder(embed_dim=embed_dim, max_len=text_max_len,
                                         num_layers=4)
        # Fusion
        self.fusion = TrimodalFusion(embed_dim)

        # Pairwise scoring head
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),   # [ref, cand, ref*cand, |ref-cand|]
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def encode_video(
        self,
        visual: torch.Tensor,         # (B, T_f, visual_dim)
        audio: torch.Tensor,          # (B, T_a, audio_dim)
        input_ids: torch.Tensor,      # (B, L_text)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a single video into a unified semantic embedding."""
        v = self.video_enc(visual)
        a = self.audio_enc(audio)
        t = self.text_enc(input_ids, attention_mask)
        return self.fusion(v, a, t)   # (B, embed_dim)

    def forward(
        self,
        # Reference video
        ref_visual: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: Optional[torch.Tensor] = None,
        # Candidate video
        cand_visual: torch.Tensor = None,
        cand_audio: torch.Tensor = None,
        cand_input_ids: torch.Tensor = None,
        cand_attention_mask: Optional[torch.Tensor] = None,
        # Target
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        ref_emb = self.encode_video(ref_visual, ref_audio, ref_input_ids, ref_attention_mask)

        if cand_visual is None:
            return {"ref_embedding": ref_emb}

        cand_emb = self.encode_video(cand_visual, cand_audio, cand_input_ids, cand_attention_mask)

        # Pairwise features: [ref, cand, element-wise product, absolute difference]
        pair_feats = torch.cat([
            ref_emb,
            cand_emb,
            ref_emb * cand_emb,
            (ref_emb - cand_emb).abs(),
        ], dim=-1)

        score = self.scorer(pair_feats).squeeze(-1)   # (B,)

        output = {
            "reproduction_score": score,
            "ref_embedding": ref_emb,
            "cand_embedding": cand_emb,
        }

        if labels is not None:
            loss = F.binary_cross_entropy(score, labels.float())
            output["loss"] = loss

        return output


# ─── MatchLite: Lightweight Student ───────────────────────────────────────────

class MatchLite(nn.Module):
    """
    MatchLite: Lightweight distilled student for production deployment.

    Compared to MatchLM:
    - Shallower encoders (fewer transformer layers)
    - Simpler fusion (concatenation instead of cross-attention)
    - ~35× cheaper inference

    Trained via knowledge distillation from MatchLM:
    - Soft target (embedding distillation): match MatchLM's embeddings
    - Hard target (label): match reproduction labels
    """

    def __init__(
        self,
        embed_dim: int = 256,           # smaller than MatchLM
        visual_input_dim: int = 512,
        audio_input_dim: int = 256,
        text_max_len: int = 128,
        teacher_embed_dim: int = 512,   # for distillation alignment
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Compact encoders (shallower)
        self.video_enc = nn.Sequential(
            nn.Linear(visual_input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.text_enc = nn.Sequential(
            nn.Embedding(32000, embed_dim),
        )
        self.text_pool = nn.AdaptiveAvgPool1d(1)

        # Simple fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Alignment projection to teacher embedding space (for distillation)
        self.align_proj = nn.Linear(embed_dim, teacher_embed_dim)

        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def encode_video(
        self,
        visual: torch.Tensor,    # (B, T_f, visual_dim) — mean-pooled in this lite version
        audio: torch.Tensor,     # (B, T_a, audio_dim)
        input_ids: torch.Tensor, # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (lite_embedding, aligned_teacher_embedding)."""
        v = self.video_enc(visual.mean(dim=1))          # (B, embed_dim)
        a = self.audio_enc(audio.mean(dim=1))           # (B, embed_dim)

        t_emb = self.text_enc[0](input_ids)             # (B, L, embed_dim)
        t = t_emb.mean(dim=1)                           # (B, embed_dim)

        fused = self.fusion_proj(torch.cat([v, a, t], dim=-1))
        aligned = self.align_proj(fused)  # project to teacher embedding space
        return fused, aligned

    def forward(
        self,
        ref_visual, ref_audio, ref_input_ids,
        cand_visual=None, cand_audio=None, cand_input_ids=None,
        labels=None,
        teacher_ref_emb=None,     # for distillation
        teacher_cand_emb=None,    # for distillation
        distill_weight: float = 0.5,
    ) -> dict:
        ref_lite, ref_aligned = self.encode_video(ref_visual, ref_audio, ref_input_ids)

        if cand_visual is None:
            return {"ref_embedding": ref_lite, "ref_aligned": ref_aligned}

        cand_lite, cand_aligned = self.encode_video(cand_visual, cand_audio, cand_input_ids)

        pair_feats = torch.cat([
            ref_lite, cand_lite,
            ref_lite * cand_lite,
            (ref_lite - cand_lite).abs(),
        ], dim=-1)
        score = self.scorer(pair_feats).squeeze(-1)

        output = {
            "reproduction_score": score,
            "ref_embedding": ref_lite,
            "cand_embedding": cand_lite,
        }

        if labels is not None:
            label_loss = F.binary_cross_entropy(score, labels.float())

            if teacher_ref_emb is not None and teacher_cand_emb is not None:
                # Embedding distillation: align student projections to teacher embeddings
                distill_loss = (
                    F.mse_loss(ref_aligned, teacher_ref_emb.detach()) +
                    F.mse_loss(cand_aligned, teacher_cand_emb.detach())
                ) * 0.5
                output["loss"] = (1 - distill_weight) * label_loss + distill_weight * distill_loss
                output["label_loss"] = label_loss.item()
                output["distill_loss"] = distill_loss.item()
            else:
                output["loss"] = label_loss

        return output
