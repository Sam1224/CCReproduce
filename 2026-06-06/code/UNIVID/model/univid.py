"""
UNIVID: Unified Vision-Language Model for Video Moderation
Paper: https://arxiv.org/abs/2606.05748
Authors: Kejuan Yang et al., ByteDance

Architecture:
    Video frames + audio transcript → UNIVID → policy-aware caption
    Caption → downstream moderation decisions (UNIVID-Lite, UNIVID-RAG, Trend Governance)

Key equations:
    Caption: c = UNIVID(V, A, P)   where V=frames, A=audio, P=policy prompt
    Risk score: r = ModerationHead(Encode(c))
    Trend embedding: e = Encoder(c) ∈ R^d
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel
from typing import Optional, List


class VideoFrameEncoder(nn.Module):
    """Encodes sampled video frames using a frozen CLIP vision encoder."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.proj = nn.Linear(768, 1024)  # project to LLM hidden dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, H, W)  T = num sampled frames
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        feats = self.encoder(pixel_values=frames_flat).last_hidden_state[:, 0]  # CLS
        feats = self.proj(feats)  # (B*T, 1024)
        return feats.view(B, T, 1024)  # (B, T, 1024)


class PolicyAwareCaptionHead(nn.Module):
    """
    Generates policy-aware captions from multimodal context.
    In the full UNIVID, this wraps an LLM backbone (e.g., Qwen-VL) with
    LoRA adapters fine-tuned on human-annotated + synthetic moderation data.
    """

    def __init__(self, lm_model_name: str = "Qwen/Qwen2.5-7B-Instruct", lora_rank: int = 16):
        super().__init__()
        # In practice: load the full LM backbone + LoRA adapters
        # Here: lightweight proxy for interface demonstration
        self.hidden_size = 1024
        self.vocab_size = 32000

        # Caption generation head (proxy — in real impl this IS the LLM)
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8, batch_first=True),
            num_layers=4
        )
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)

        # LoRA-style delta matrices (simulated)
        self.lora_A = nn.Linear(self.hidden_size, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, self.hidden_size, bias=False)
        self.lora_scale = 0.1

    def forward(
        self,
        visual_features: torch.Tensor,      # (B, T, hidden)
        text_query: torch.Tensor,           # (B, L_q, hidden)  policy query embedding
        caption_ids: Optional[torch.Tensor] = None,  # (B, L_c) for teacher forcing
    ) -> torch.Tensor:
        # Fuse visual features as memory for the cross-attention in TransformerDecoder
        memory = torch.cat([visual_features, text_query], dim=1)  # (B, T+L_q, hidden)

        if caption_ids is not None:
            # Teacher-forced training mode
            tgt = text_query  # reuse as target proxy
            out = self.caption_decoder(tgt, memory)
            # Apply LoRA delta
            out = out + self.lora_scale * self.lora_B(self.lora_A(out))
            logits = self.output_proj(out)  # (B, L_c, vocab_size)
            return logits
        else:
            # Greedy decoding proxy (real impl uses beam search / sampling in the LLM)
            tgt = text_query[:, :1, :]  # start token
            for _ in range(50):  # max caption tokens
                out = self.caption_decoder(tgt, memory)
                out = out + self.lora_scale * self.lora_B(self.lora_A(out))
                next_logit = self.output_proj(out[:, -1:, :])
                next_id = next_logit.argmax(dim=-1, keepdim=True)  # greedy
                # In real impl: append token to tgt and check EOS
                # Simplified: just return logit sequences
            return self.output_proj(out)


class CaptionEncoder(nn.Module):
    """Encodes a generated caption into a dense embedding for similarity/clustering."""

    def __init__(self, hidden_size: int = 1024, embed_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, caption_features: torch.Tensor) -> torch.Tensor:
        # caption_features: (B, L, hidden)
        pooled = self.pool(caption_features.transpose(1, 2)).squeeze(-1)  # (B, hidden)
        return nn.functional.normalize(self.proj(pooled), dim=-1)  # (B, embed_dim)


class UNIVID(nn.Module):
    """
    Full UNIVID model.

    Pipeline:
        1. Encode video frames with VideoFrameEncoder
        2. Generate policy-aware caption via PolicyAwareCaptionHead
        3. Encode caption to embedding for downstream use (UNIVID-RAG, Trend Governance)
    """

    def __init__(
        self,
        num_frames: int = 8,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.frame_encoder = VideoFrameEncoder()
        self.caption_head = PolicyAwareCaptionHead()
        self.caption_encoder = CaptionEncoder(embed_dim=embed_dim)

        # Policy query embedding table (one learned vector per moderation category)
        self.policy_embeddings = nn.Embedding(64, 1024)  # 64 policy categories

    def encode_policy(self, policy_ids: torch.Tensor) -> torch.Tensor:
        """Returns dense policy query vectors."""
        return self.policy_embeddings(policy_ids)  # (B, num_policies, 1024)

    def forward(
        self,
        frames: torch.Tensor,           # (B, T, 3, H, W)
        policy_ids: torch.Tensor,       # (B, P)  policy category indices
        caption_ids: Optional[torch.Tensor] = None,  # for training
    ) -> dict:
        visual_features = self.frame_encoder(frames)  # (B, T, 1024)
        policy_vectors = self.encode_policy(policy_ids)  # (B, P, 1024)

        caption_logits = self.caption_head(
            visual_features, policy_vectors, caption_ids
        )  # (B, L, vocab_size)

        caption_features = self.caption_head.lora_B(
            self.caption_head.lora_A(visual_features)
        )  # proxy for caption hidden states
        caption_embedding = self.caption_encoder(caption_features)  # (B, embed_dim)

        return {
            "caption_logits": caption_logits,
            "caption_embedding": caption_embedding,
        }
