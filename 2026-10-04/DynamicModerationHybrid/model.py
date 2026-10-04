from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    visual_dim: int = 64
    audio_dim: int = 32
    text_dim: int = 48
    hidden_dim: int = 128
    embedding_dim: int = 96
    num_classes: int = 5


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalStudent(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.visual = ModalityProjector(config.visual_dim, config.hidden_dim)
        self.audio = ModalityProjector(config.audio_dim, config.hidden_dim)
        self.text = ModalityProjector(config.text_dim, config.hidden_dim)
        self.gate = nn.Sequential(nn.Linear(config.hidden_dim * 3, 3), nn.Softmax(dim=-1))
        self.embedding_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)
        self.reranker = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual = self.visual(batch["visual"].float())
        audio = self.audio(batch["audio"].float())
        text = self.text(batch["text"].float())
        stacked = torch.stack([visual, audio, text], dim=1)
        gates = self.gate(torch.cat([visual, audio, text], dim=-1)).unsqueeze(-1)
        fused = (stacked * gates).sum(dim=1)
        return F.normalize(self.embedding_head(fused), dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedding = self.encode(batch)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}

    def score_references(self, clip_embedding: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, num_refs = clip_embedding.size(0), reference_embeddings.size(0)
        clip = clip_embedding[:, None, :].expand(batch_size, num_refs, -1)
        ref = reference_embeddings[None, :, :].expand(batch_size, num_refs, -1)
        cosine = F.cosine_similarity(clip, ref, dim=-1, eps=1e-6).unsqueeze(-1)
        features = torch.cat([clip, ref, cosine], dim=-1)
        return self.reranker(features).squeeze(-1)


class FrozenMLLMTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        raw_dim = config.visual_dim + config.audio_dim + config.text_dim
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, config.hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 2, config.embedding_dim),
        )
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw = torch.cat([batch["visual"].float(), batch["audio"].float(), batch["text"].float()], dim=-1)
        embedding = F.normalize(self.encoder(raw), dim=-1)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}


def distillation_loss(student: Dict[str, torch.Tensor], teacher: Dict[str, torch.Tensor], temperature: float = 2.0) -> torch.Tensor:
    soft_targets = F.softmax(teacher["logits"] / temperature, dim=-1)
    soft_log_probs = F.log_softmax(student["logits"] / temperature, dim=-1)
    kl = F.kl_div(soft_log_probs, soft_targets, reduction="batchmean") * temperature * temperature
    embed = 1.0 - F.cosine_similarity(student["embedding"], teacher["embedding"], dim=-1).mean()
    return kl + embed


def contrastive_reference_loss(scores: torch.Tensor, reference_ids: torch.Tensor) -> torch.Tensor:
    targets = reference_ids.clamp(max=scores.size(1) - 1)
    return F.cross_entropy(scores, targets)


def hybrid_decision(class_probs: torch.Tensor, reference_scores: torch.Tensor, class_threshold: float = 0.62, sim_threshold: float = 0.55) -> torch.Tensor:
    violation_prob = 1.0 - class_probs[:, 0]
    sim_prob = torch.sigmoid(reference_scores.max(dim=1).values)
    return (violation_prob >= class_threshold) | (sim_prob >= sim_threshold)
