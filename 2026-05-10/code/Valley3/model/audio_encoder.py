"""
Valley3 Audio Encoder stub.
In production: replace with Whisper encoder (openai/whisper-medium or large-v3).

Valley3's key differentiator: native multilingual audio for e-commerce short videos.
Supports Chinese (Mandarin/Cantonese), English, Indonesian, Arabic, etc.
"""

import torch
import torch.nn as nn


class AudioEncoderStub(nn.Module):
    """
    Stub audio encoder for testing.
    Production: Whisper-large-v3 encoder (1500 → 1500 token positions)

    Input: mel spectrogram [B, 80, T] where T=3000 for 30-second audio
    Output: audio features [B, T//2, hidden_size] after Conv downsampling
    """

    def __init__(self, hidden_size: int = 1024, mel_bins: int = 80):
        super().__init__()
        self.hidden_size = hidden_size
        # Whisper-style: 2x Conv1d layers for 4x temporal downsampling
        self.conv1 = nn.Conv1d(mel_bins, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [B, mel_bins, T]
        Returns:
            [B, hidden_size, T//2]  (channels-first for Conv1d compatibility)
        """
        x = self.act(self.conv1(mel_spectrogram))  # [B, hidden, T]
        x = self.act(self.conv2(x))                 # [B, hidden, T//2]
        return x
