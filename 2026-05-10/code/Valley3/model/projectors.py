"""
Cross-modal projectors for Valley3.
Maps vision/audio encoded features to LLM embedding space.
"""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """
    Two-layer MLP projector: vision tokens → LLM embedding space.
    Valley3 paper uses MLP projector (common in LLaVA-style models).
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AudioProjector(nn.Module):
    """
    Audio projector with temporal compression.
    Valley3's native multilingual audio capability for short-video e-commerce.

    Compresses audio token sequence (Whisper encoder outputs long sequences)
    before projecting to LLM space.
    """

    def __init__(self, audio_hidden_size: int, llm_hidden_size: int, compression_factor: int = 2):
        super().__init__()
        self.compression_factor = compression_factor
        # 1D Conv for temporal compression (halves sequence length)
        self.compress = nn.Conv1d(
            audio_hidden_size,
            audio_hidden_size,
            kernel_size=compression_factor,
            stride=compression_factor,
            padding=0,
        )
        self.proj = nn.Sequential(
            nn.Linear(audio_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [B, audio_hidden_size, T] (Conv1d expects channels first)
        Returns:
            [B, T//compression_factor, llm_hidden_size]
        """
        compressed = self.compress(audio_features)   # [B, audio_hidden, T//2]
        compressed = compressed.transpose(1, 2)       # [B, T//2, audio_hidden]
        return self.proj(compressed)
