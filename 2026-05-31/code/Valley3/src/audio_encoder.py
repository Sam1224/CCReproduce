"""
Audio Transformer for Valley3 (simplified toy implementation).

Paper: Valley3 extends Qwen3-VL with an Audio Transformer that encodes
Mel spectrogram features, aligned to the visual-language backbone via
an MLP connector.

Architecture (from paper Section 3.1):
  Audio signal → Mel Spectrogram → Audio Transformer → Audio embeddings
  Audio embeddings → MLP Connector → aligned with visual/text token space
"""
import torch
import torch.nn as nn
import math


class AudioTransformerEncoder(nn.Module):
    """
    Simplified Audio Transformer for encoding Mel spectrograms.
    Real Valley3 uses a full audio backbone (e.g., Whisper-like architecture).
    """
    def __init__(self, mel_bins: int = 80, max_frames: int = 128,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 output_dim: int = 512):
        super().__init__()
        self.mel_proj = nn.Linear(mel_bins, d_model)
        self.pos_emb = nn.Embedding(max_frames, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram [B, T, mel_bins]
        Returns:
            audio_embedding: [B, output_dim]
        """
        B, T, _ = mel.shape
        x = self.mel_proj(mel)
        pos_ids = torch.arange(T, device=mel.device)
        x = x + self.pos_emb(pos_ids).unsqueeze(0)
        x = self.transformer(x)  # [B, T, d_model]
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        return self.output_proj(x)  # [B, output_dim]


class AudioMLPConnector(nn.Module):
    """MLP connector that aligns audio embeddings with LLM token space.
    Paper: audio embeddings are aligned via MLP and concatenated with
    visual/text tokens into a unified input sequence.
    """
    def __init__(self, audio_dim: int = 512, llm_dim: int = 512, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim * num_tokens),
        )

    def forward(self, audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_emb: [B, audio_dim]
        Returns:
            audio_tokens: [B, num_tokens, llm_dim]
        """
        B = audio_emb.shape[0]
        out = self.mlp(audio_emb)                       # [B, llm_dim * num_tokens]
        return out.view(B, self.num_tokens, -1)         # [B, num_tokens, llm_dim]
