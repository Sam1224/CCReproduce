"""
MLLM-based safety judge for PaSBench-Video evaluation.

Wraps any HuggingFace-compatible multimodal LLM to serve as a
streaming safety judge. Also includes a RandomJudge baseline
and a SimpleMotionJudge for ablation.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.base_judge import BaseVideoSafetyJudge, StreamingPrediction


class RandomJudge(BaseVideoSafetyJudge):
    """
    Random baseline: issues warning at random frame with probability p_warn_per_frame.
    Expected to achieve ~20% recall with ~20% FPR.
    """

    def __init__(self, p_warn_per_frame: float = 0.05, seed: int = 42):
        self.p = p_warn_per_frame
        random.seed(seed)

    def predict_streaming(
        self, frames: torch.Tensor, question: str, frame_index: int
    ) -> StreamingPrediction:
        issue = random.random() < self.p
        return StreamingPrediction(
            issue_warning=issue,
            warning_text="Potential safety risk detected." if issue else None,
            confidence=random.random(),
        )


class SimpleMotionJudge(BaseVideoSafetyJudge):
    """
    Ablation baseline: detects sudden motion changes as proxy for risk onset.
    Uses frame difference as a simple feature.

    This is a strong heuristic baseline for temporal calibration.
    """

    def __init__(self, motion_threshold: float = 0.15, min_frame: int = 5):
        self.threshold = motion_threshold
        self.min_frame = min_frame
        self.prev_frame: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.prev_frame = None

    def predict_streaming(
        self, frames: torch.Tensor, question: str, frame_index: int
    ) -> StreamingPrediction:
        if frame_index < self.min_frame or frames.shape[0] < 2:
            self.prev_frame = frames[-1]
            return StreamingPrediction(issue_warning=False, warning_text=None, confidence=0.0)

        curr = frames[-1].float()
        prev = frames[-2].float()
        motion = (curr - prev).abs().mean().item()

        issue = motion > self.threshold
        return StreamingPrediction(
            issue_warning=issue,
            warning_text=f"Sudden motion detected (score={motion:.3f})" if issue else None,
            confidence=min(motion / self.threshold, 1.0),
        )


class TemporalSafetyTransformer(nn.Module):
    """
    Lightweight temporal transformer for safety prediction.

    Architecture:
    - Frame feature extractor (simplified CNN or pretrained backbone)
    - Temporal transformer encoder with causal attention mask
    - Safety classification head

    This is a training-capable version for fine-tuning on safety data.
    In PaSBench-Video evaluation, we assess zero-shot performance of
    large MLLMs; this model serves as a trainable baseline.

    Input: frames (B, T, C, H, W)
    Output: per-frame safety logits (B, T, 2)  [no-risk, risk]
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # Simple patch embedding
        self.patch_embed = nn.Sequential(
            nn.Unflatten(-1, (img_size // patch_size, patch_size, img_size // patch_size, patch_size)),
            # Will be handled in forward
        )
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.frame_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # Frame-level transformer (spatial)
        frame_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.frame_transformer = nn.TransformerEncoder(frame_layer, num_layers=2)

        # Temporal transformer (causal)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        self.temporal_pos_embed = nn.Embedding(512, d_model)

        # Safety classification head
        self.classifier = nn.Linear(d_model, 2)
        self.patch_size = patch_size
        self.img_size = img_size

    def _extract_patches(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B*T, C, H, W)
        Returns: (B*T, num_patches, patch_dim)
        """
        B, C, H, W = frames.shape
        p = self.patch_size
        # Reshape to patches
        x = frames.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H/p, W/p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode a single frame using spatial transformer.
        frame: (B, C, H, W)
        Returns: (B, d_model)  — CLS token embedding
        """
        B = frame.shape[0]
        patches = self._extract_patches(frame)  # (B, N, patch_dim)
        tokens = self.patch_proj(patches)  # (B, N, d_model)

        cls = self.frame_cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, d_model)
        tokens = tokens + self.frame_pos_embed[:, : tokens.shape[1], :]

        tokens = self.frame_transformer(tokens)  # (B, N+1, d_model)
        return tokens[:, 0]  # CLS token: (B, d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        Returns: (B, T, 2)  — per-frame safety logits

        Uses causal attention mask so each position only attends to prior frames.
        """
        B, T, C, H, W = frames.shape

        # Encode each frame
        frames_flat = frames.reshape(B * T, C, H, W)
        frame_embeds = self.encode_frame(frames_flat)  # (B*T, d_model)
        frame_embeds = frame_embeds.reshape(B, T, self.d_model)

        # Add temporal positional encoding
        pos_ids = torch.arange(T, device=frames.device)
        frame_embeds = frame_embeds + self.temporal_pos_embed(pos_ids).unsqueeze(0)

        # Causal mask (upper triangle = -inf so future frames are not attended to)
        causal_mask = torch.triu(
            torch.ones(T, T, device=frames.device) * float("-inf"), diagonal=1
        )

        temporal_out = self.temporal_transformer(
            frame_embeds, mask=causal_mask
        )  # (B, T, d_model)

        logits = self.classifier(temporal_out)  # (B, T, 2)
        return logits


class TransformerJudge(BaseVideoSafetyJudge):
    """
    Streaming judge based on TemporalSafetyTransformer.
    Issues warning when risk probability exceeds threshold.
    """

    def __init__(
        self,
        model: Optional[TemporalSafetyTransformer] = None,
        threshold: float = 0.5,
        device: str = "cpu",
        min_warning_frame: int = 3,
    ):
        if model is None:
            model = TemporalSafetyTransformer()
        self.model = model.to(device).eval()
        self.threshold = threshold
        self.device = device
        self.min_warning_frame = min_warning_frame
        self._frame_buffer = []

    def reset(self) -> None:
        self._frame_buffer = []

    def predict_streaming(
        self,
        frames: torch.Tensor,
        question: str,
        frame_index: int,
    ) -> StreamingPrediction:
        if frame_index < self.min_warning_frame:
            return StreamingPrediction(issue_warning=False, warning_text=None, confidence=0.0)

        # Add batch dimension: (1, T, C, H, W)
        frames_in = frames.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(frames_in)  # (1, T, 2)
            probs = F.softmax(logits[0, -1], dim=-1)  # (2,) for current frame
            risk_prob = probs[1].item()

        issue = risk_prob >= self.threshold
        return StreamingPrediction(
            issue_warning=issue,
            warning_text=f"Risk detected (p={risk_prob:.3f})" if issue else None,
            confidence=risk_prob,
        )
