"""
GeminiEmbedding2 — faithful PyTorch reproduction of the core architecture.

Paper: "Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini"
arXiv: 2605.27295  (submitted 2026-05-26)

Architecture summary:
  - Shared transformer backbone (decoder-style, like Gemini)
  - Late-layer hidden states → mean-pooled → L2-normalized embedding
  - Modality-specific tokenizers feed into a unified token sequence
  - Training: multi-task multi-stage contrastive (InfoNCE) loss

This reproduction implements:
  1. GeminiEmbeddingModel  — backbone + projection + pooling
  2. ModalityTokenizer      — text/image/audio/video input encoding
  3. MultiTaskInfoNCELoss   — multi-task contrastive loss
  4. Training + evaluation scripts (train.py, evaluate.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GE2Config:
    # Backbone (toy-scale defaults; real model uses Gemini scale)
    vocab_size: int = 32_000
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Embedding head
    embed_dim: int = 768          # final embedding dimension
    num_last_layers: int = 4      # pool over last N layers
    normalize: bool = True        # L2-normalize final embedding

    # Modality token sizes (toy defaults)
    image_patch_size: int = 16    # patches per side (image_patch_size^2 patches total)
    image_size: int = 224
    audio_mel_bins: int = 80
    audio_frame_len: int = 128
    video_frames: int = 8         # sampled frames per video


# ---------------------------------------------------------------------------
# Backbone: simplified Gemini-style decoder transformer (encoder-mode)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, T, D)
        v = v.transpose(1, 2)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rotary(q, k, cos, sin)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation (Gemini-style)
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.hidden_dim)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.hidden_dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Modality tokenizers
# ---------------------------------------------------------------------------

class TextTokenizer(nn.Module):
    """Standard token embedding for text input."""
    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class ImageTokenizer(nn.Module):
    """Patch embedding for images (ViT-style)."""
    def __init__(self, cfg: GE2Config):
        super().__init__()
        num_patches = (cfg.image_size // cfg.image_patch_size) ** 2
        patch_dim = 3 * cfg.image_patch_size ** 2
        self.proj = nn.Linear(patch_dim, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(num_patches + 1, cfg.hidden_dim)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, patch_dim)  — already extracted patches
        """
        B, N, _ = patches.shape
        x = self.proj(patches)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = self.pos_embed(torch.arange(N + 1, device=patches.device))
        return x + pos


class AudioTokenizer(nn.Module):
    """Simple 1D conv tokenizer for mel spectrograms."""
    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.audio_mel_bins, cfg.hidden_dim, kernel_size=16, stride=8, padding=4),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, mel_bins, T)"""
        return self.conv(mel).transpose(1, 2)  # (B, T', hidden_dim)


class VideoTokenizer(nn.Module):
    """Per-frame image tokenizer + temporal position."""
    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.frame_tokenizer = ImageTokenizer(cfg)
        self.temporal_embed = nn.Embedding(cfg.video_frames, cfg.hidden_dim)

    def forward(self, frames: torch.Tensor, frame_patches: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, F)  — frame indices for temporal position
        frame_patches: (B, F, N, patch_dim)  — patches per frame
        """
        B, F, N, D = frame_patches.shape
        flat_patches = frame_patches.reshape(B * F, N, D)
        tokens = self.frame_tokenizer(flat_patches)  # (B*F, N+1, hidden_dim)
        tokens = tokens.reshape(B, F, -1, tokens.shape[-1])
        temp = self.temporal_embed(frames).unsqueeze(2)  # (B, F, 1, hidden_dim)
        return (tokens + temp).reshape(B, F * tokens.shape[2], -1)


# ---------------------------------------------------------------------------
# Main embedding model
# ---------------------------------------------------------------------------

class GeminiEmbeddingModel(nn.Module):
    """
    Gemini Embedding 2 — native multimodal embedding model.

    Input: sequence of tokens from any combination of text/image/audio/video.
    Output: L2-normalized embedding vector of shape (B, embed_dim).

    Key design choices (from paper):
      - Shared transformer backbone for all modalities
      - Average hidden states from last `num_last_layers` layers
      - Mean pool over token dimension → projection → L2 normalize
    """

    def __init__(self, cfg: GE2Config):
        super().__init__()
        self.cfg = cfg

        # Modality tokenizers
        self.text_tok = TextTokenizer(cfg)
        self.image_tok = ImageTokenizer(cfg)
        self.audio_tok = AudioTokenizer(cfg)
        self.video_tok = VideoTokenizer(cfg)

        # Modality-type embedding (distinguishes modalities in sequence)
        self.modality_embed = nn.Embedding(5, cfg.hidden_dim)  # 0=pad, 1=text, 2=img, 3=audio, 4=video

        # Shared backbone
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_dim)

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns token embeddings for text input."""
        tokens = self.text_tok(input_ids)
        mod_emb = self.modality_embed(torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device))
        return tokens + mod_emb

    def encode_image(self, patches: torch.Tensor) -> torch.Tensor:
        """Returns token embeddings for image patches."""
        B, N, D = patches.shape
        tokens = self.image_tok(patches)
        mod_ids = torch.full((B, tokens.shape[1]), 2, dtype=torch.long, device=patches.device)
        return tokens + self.modality_embed(mod_ids)

    def forward_backbone(self, tokens: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Run backbone and return hidden states from each layer."""
        x = tokens
        hidden_states = []
        # Build causal-ish mask (bidirectional for embedding, no causal mask)
        if attention_mask is not None:
            # Convert padding mask (B, T) to additive attention mask
            mask = attention_mask[:, None, None, :].float()
            mask = (1.0 - mask) * -1e9
        else:
            mask = None
        for block in self.blocks:
            x = block(x, mask)
            hidden_states.append(x)
        return hidden_states

    def pool_and_project(self, hidden_states: List[torch.Tensor],
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Average last `num_last_layers` layers → mean pool → project → L2 normalize.
        """
        # Average over last N layers
        last_n = hidden_states[-self.cfg.num_last_layers:]
        pooled_layers = torch.stack(last_n, dim=0).mean(dim=0)  # (B, T, hidden_dim)
        pooled_layers = self.norm(pooled_layers)

        # Mean pool over token dimension (masked)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embed = (pooled_layers * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            embed = pooled_layers.mean(dim=1)

        # Project and normalize
        embed = self.proj(embed)
        if self.cfg.normalize:
            embed = F.normalize(embed, p=2, dim=-1)
        return embed

    def forward(self, tokens: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tokens: (B, T, hidden_dim) — pre-encoded token embeddings from any modality
        Returns: (B, embed_dim) — L2-normalized embedding
        """
        hidden_states = self.forward_backbone(tokens, attention_mask)
        return self.pool_and_project(hidden_states, attention_mask)


# ---------------------------------------------------------------------------
# Multi-task InfoNCE loss
# ---------------------------------------------------------------------------

class MultiTaskInfoNCELoss(nn.Module):
    """
    Multi-task contrastive loss (Section 3.2 of Gemini Embedding paper).

    For each task t, compute InfoNCE between query embeddings q and
    positive key embeddings k+. Hard negatives from in-batch negatives.
    Task weights are learned or fixed.

    Paper uses multi-stage training: first broad web-scale pairs (text-image,
    text-text), then domain-specific tasks.
    """

    def __init__(self, temperature: float = 0.02, num_tasks: int = 4):
        super().__init__()
        self.temperature = temperature
        # Learnable per-task temperature scalings
        self.task_temp = nn.Parameter(torch.ones(num_tasks) * temperature)

    def info_nce(self, q: torch.Tensor, k: torch.Tensor,
                 task_idx: int = 0) -> torch.Tensor:
        """
        Symmetric InfoNCE loss.
        q, k: (B, D) L2-normalized embeddings.
        Positives: diagonal pairs. Negatives: all off-diagonal pairs in batch.
        """
        B = q.size(0)
        temp = self.task_temp[task_idx].clamp(min=0.005, max=0.5)
        logits = torch.matmul(q, k.T) / temp  # (B, B)
        labels = torch.arange(B, device=q.device)
        loss_q = F.cross_entropy(logits, labels)
        loss_k = F.cross_entropy(logits.T, labels)
        return (loss_q + loss_k) / 2.0

    def forward(self, embeddings: Dict[str, torch.Tensor],
                task_assignments: torch.Tensor) -> torch.Tensor:
        """
        embeddings: {"query": (B, D), "key": (B, D)}
        task_assignments: (B,) — task index per sample
        """
        q = embeddings["query"]
        k = embeddings["key"]
        total_loss = torch.tensor(0.0, device=q.device)
        unique_tasks = task_assignments.unique()
        for t in unique_tasks:
            mask = task_assignments == t
            if mask.sum() < 2:
                continue
            q_t = q[mask]
            k_t = k[mask]
            total_loss += self.info_nce(q_t, k_t, task_idx=t.item())
        return total_loss / max(len(unique_tasks), 1)
