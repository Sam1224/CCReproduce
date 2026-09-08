from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from data import IMAGE_DIM, VOCAB_SIZE, EXPANSION_VOCAB


class TitleEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(title_ids)
        pooled = embedded.mean(dim=1)
        return self.proj(pooled)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMAGE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.net(image_features)


class SAMD2QModel(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.title_encoder = TitleEncoder(hidden_dim=hidden_dim)
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(EXPANSION_VOCAB)),
        )

    def forward(self, title_ids: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        title_repr = self.title_encoder(title_ids)
        image_repr = self.image_encoder(image_features)
        fused = torch.cat([title_repr, image_repr], dim=-1)
        return self.head(fused)


class RewardAlignmentLoss(nn.Module):
    def forward(self, logits: torch.Tensor, reward_targets: torch.Tensor) -> torch.Tensor:
        teacher_distribution = reward_targets / reward_targets.sum(dim=-1, keepdim=True)
        student_log_prob = F.log_softmax(logits, dim=-1)
        return F.kl_div(student_log_prob, teacher_distribution, reduction="batchmean")


def token_recall_at_k(logits: torch.Tensor, positive_mask: torch.Tensor, k: int = 3) -> float:
    topk = logits.topk(k=min(k, logits.size(-1)), dim=-1).indices
    hits = []
    for row_index, predicted in enumerate(topk):
        positives = positive_mask[row_index].nonzero(as_tuple=False).flatten().tolist()
        predicted_set = set(predicted.tolist())
        if not positives:
            hits.append(1.0)
            continue
        overlap = len(predicted_set.intersection(positives)) / len(positives)
        hits.append(overlap)
    return float(sum(hits) / len(hits))


def expansion_tokens_from_logits(logits: torch.Tensor, k: int = 2) -> list[list[str]]:
    topk = logits.topk(k=min(k, logits.size(-1)), dim=-1).indices.cpu().tolist()
    return [[EXPANSION_VOCAB[index] for index in row] for row in topk]
