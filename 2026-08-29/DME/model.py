from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TextTower(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).float().unsqueeze(-1)
        token_states = self.embedding(token_ids)
        pooled = (token_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class FrameTower(nn.Module):
    def __init__(self, frame_dim: int = 10, hidden_dim: int = 128) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(nn.Linear(frame_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.temporal_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frame_states = self.frame_encoder(frames)
        weights = torch.softmax(self.temporal_gate(frame_states).squeeze(-1), dim=1)
        return (frame_states * weights.unsqueeze(-1)).sum(dim=1)


class DMEModel(nn.Module):
    def __init__(self, vocab_size: int, frame_dim: int = 10, hidden_dim: int = 128, temperature: float = 0.07) -> None:
        super().__init__()
        self.query_tower = TextTower(vocab_size, hidden_dim)
        self.text_tower = TextTower(vocab_size, hidden_dim)
        self.frame_tower = FrameTower(frame_dim, hidden_dim)
        self.video_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.temperature = temperature

    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.query_tower(query_ids), dim=-1)

    def encode_video(self, text_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        text_state = self.text_tower(text_ids)
        frame_state = self.frame_tower(frames)
        return functional.normalize(self.video_fusion(torch.cat([text_state, frame_state], dim=-1)), dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_embedding = self.encode_query(batch["query_ids"])
        video_embedding = self.encode_video(batch["text_ids"], batch["frames"])
        return functional.cosine_similarity(query_embedding, video_embedding) / self.temperature

    def loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        logits = self.forward(batch)
        labels = batch["label"]
        binary_loss = functional.binary_cross_entropy_with_logits(logits, labels)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (predictions == labels).float().mean()
        return binary_loss, {"loss": float(binary_loss.detach().cpu()), "accuracy": float(accuracy.detach().cpu())}
