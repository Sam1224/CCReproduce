"""
Valley3: Omni Multimodal Large Language Model for E-commerce
Reproduces the core architecture from arXiv 2605.01278.

Architecture overview:
  Input: text / image / video / audio (omni)
  → Modality-specific encoders
  → Cross-modal projector (MLP)
  → LLM backbone (e.g., Qwen2.5 / LLaMA-3)
  → Output: text tokens (+ reasoning chain if thinking mode)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class Valley3Config:
    llm_hidden_size: int = 4096
    vision_hidden_size: int = 1024
    audio_hidden_size: int = 512
    text_vocab_size: int = 151936  # Qwen2.5 vocab
    max_seq_len: int = 32768
    # Controllable reasoning modes: 0=no-think, 1/2/3=increasing depth
    reasoning_mode: Literal[0, 1, 2, 3] = 0
    num_heads: int = 32
    num_layers: int = 32


class VisionEncoder(nn.Module):
    """
    Simplified vision encoder (CLIP/SigLIP style).
    Full Valley3 uses a pretrained CLIP-L or SigLIP encoder.
    """
    def __init__(self, cfg: Valley3Config):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, cfg.vision_hidden_size, kernel_size=14, stride=14)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, cfg.vision_hidden_size) * 0.02)
        self.norm = nn.LayerNorm(cfg.vision_hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        B = pixel_values.size(0)
        x = self.patch_embed(pixel_values)              # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)               # [B, N, D]
        x = x + self.pos_embed[:, :x.size(1)]
        return self.norm(x)


class AudioEncoder(nn.Module):
    """
    Multilingual audio encoder for e-commerce livestream audio.
    Full Valley3 uses Whisper-large or equivalent as backbone.
    Stage 1 pre-training aligns audio representations with LLM space.
    """
    def __init__(self, cfg: Valley3Config):
        super().__init__()
        # Mel-spectrogram feature extractor (simplified)
        self.conv1 = nn.Conv1d(80, cfg.audio_hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.audio_hidden_size, cfg.audio_hidden_size, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(cfg.audio_hidden_size)
        self.gelu = nn.GELU()

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        # mel_features: [B, 80, T] — 80-dim mel spectrogram
        x = self.gelu(self.conv1(mel_features))
        x = self.gelu(self.conv2(x))         # [B, D, T//2]
        x = x.transpose(1, 2)               # [B, T//2, D]
        return self.norm(x)


class CrossModalProjector(nn.Module):
    """
    2-layer MLP projector aligning vision/audio features to LLM space.
    Valley3 paper §3.2: separate projectors for vision and audio.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ControllableReasoningHead(nn.Module):
    """
    Controllable reasoning mode controller.
    Valley3 post-training introduces 4 modes:
      mode=0: non-thinking (fast, for simple tasks)
      mode=1: light thinking
      mode=2: moderate thinking
      mode=3: deep thinking (for complex violation analysis)

    Mechanism: prepend a mode-specific system token that conditions the
    LLM's chain-of-thought budget via trained control tokens.
    """
    def __init__(self, cfg: Valley3Config, tokenizer_vocab_size: int):
        super().__init__()
        # 4 learnable control token embeddings (one per mode)
        self.mode_embeddings = nn.Embedding(4, cfg.llm_hidden_size)
        self.mode = cfg.reasoning_mode

    def get_mode_token(self) -> torch.Tensor:
        idx = torch.tensor([self.mode])
        return self.mode_embeddings(idx)  # [1, D]

    def set_mode(self, mode: int):
        assert 0 <= mode <= 3
        self.mode = mode


class Valley3Model(nn.Module):
    """
    Full Valley3 omni MLLM for e-commerce.

    Four-stage pre-training pipeline (Valley3 §3):
      Stage 1: Audio understanding — align audio encoder to LLM space
      Stage 2: Cross-modal instruction following — joint text/image/audio IT
      Stage 3: E-commerce domain — product understanding, livestream, governance
      Stage 4: Long-context reasoning — multi-turn e-com deep research

    Post-training:
      - Controllable reasoning (SFT on thinking-mode data)
      - Agentic search integration (tool-use SFT)
    """

    def __init__(self, cfg: Valley3Config):
        super().__init__()
        self.cfg = cfg

        # Modality encoders
        self.vision_encoder = VisionEncoder(cfg)
        self.audio_encoder = AudioEncoder(cfg)

        # Cross-modal projectors (align to LLM hidden dim)
        self.vision_projector = CrossModalProjector(cfg.vision_hidden_size, cfg.llm_hidden_size)
        self.audio_projector = CrossModalProjector(cfg.audio_hidden_size, cfg.llm_hidden_size)

        # LLM backbone (simplified Transformer decoder)
        # Full implementation uses Qwen2.5-7B or LLaMA-3-8B as backbone
        self.embedding = nn.Embedding(cfg.text_vocab_size, cfg.llm_hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=cfg.llm_hidden_size,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.llm_hidden_size * 4,
                batch_first=True,
            )
            for _ in range(cfg.num_layers)
        ])
        self.lm_head = nn.Linear(cfg.llm_hidden_size, cfg.text_vocab_size, bias=False)

        # Controllable reasoning
        self.reasoning_head = ControllableReasoningHead(cfg, cfg.text_vocab_size)

    def encode_vision(self, pixel_values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pixel_values is None:
            return None
        feats = self.vision_encoder(pixel_values)
        return self.vision_projector(feats)

    def encode_audio(self, mel_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mel_features is None:
            return None
        feats = self.audio_encoder(mel_features)
        return self.audio_projector(feats)

    def encode_video(self, video_frames: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Video is processed as a sequence of frames.
        Full Valley3 uses temporal position embeddings + frame compression.
        # TODO: implement frame compression (e.g., average pooling per N frames)
        """
        if video_frames is None:
            return None
        B, T, C, H, W = video_frames.shape
        frames = video_frames.view(B * T, C, H, W)
        feats = self.vision_encoder(frames)             # [B*T, N, D]
        feats = feats.view(B, T * feats.size(1), -1)   # [B, T*N, D]
        return self.vision_projector(feats)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        mel_features: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Unified forward pass for all modality combinations.

        Multi-modal token merging (Valley3 §3.1):
          [mode_token] [vision_tokens] [audio_tokens] [text_tokens]
        """
        # Text embeddings
        text_embeds = self.embedding(input_ids)  # [B, L, D]

        # Prepend reasoning mode token
        mode_token = self.reasoning_head.get_mode_token()  # [1, D]
        mode_token = mode_token.unsqueeze(0).expand(text_embeds.size(0), -1, -1)

        # Concatenate multimodal tokens
        input_embeds = [mode_token, text_embeds]

        vision_embeds = self.encode_vision(pixel_values)
        if vision_embeds is not None:
            input_embeds.insert(1, vision_embeds)

        audio_embeds = self.encode_audio(mel_features)
        if audio_embeds is not None:
            input_embeds.insert(1, audio_embeds)

        video_embeds = self.encode_video(video_frames)
        if video_embeds is not None:
            input_embeds.insert(1, video_embeds)

        hidden_states = torch.cat(input_embeds, dim=1)  # [B, total_len, D]

        # Simplified causal mask (full: RoPE + Flash Attention)
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, hidden_states)

        logits = self.lm_head(hidden_states)  # [B, total_len, V]

        loss = None
        if labels is not None:
            # Shift for causal LM loss — align with text portion only
            text_start = hidden_states.size(1) - input_ids.size(1)
            text_logits = logits[:, text_start:, :]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                text_logits.view(-1, self.cfg.text_vocab_size),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}
