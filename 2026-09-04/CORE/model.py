from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from data import FEATURE_DIM, VOCAB_SIZE


class QueryEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE + 1, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embed(query_tokens)
        pooled = embedded.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


class CandidateEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        encoded = self.net(candidate_features)
        return F.normalize(encoded, dim=-1)


class COREEmbed(nn.Module):
    def __init__(self, hidden_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.query_encoder = QueryEncoder(hidden_dim=hidden_dim)
        self.candidate_encoder = CandidateEncoder(hidden_dim=hidden_dim)
        self.temperature = temperature

    def forward(self, query_tokens: torch.Tensor, candidate_features: torch.Tensor):
        query_repr = self.query_encoder(query_tokens)
        candidate_repr = self.candidate_encoder(candidate_features)
        scores = torch.einsum("bd,bkd->bk", query_repr, candidate_repr) / self.temperature
        return scores


class RankKLLoss(nn.Module):
    def forward(self, student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        student_log_prob = F.log_softmax(student_scores, dim=-1)
        teacher_prob = F.softmax(teacher_scores, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def retrieval_metrics(scores: torch.Tensor, targets: torch.Tensor) -> dict:
    top1 = scores.argmax(dim=-1)
    acc = (top1 == targets).float().mean().item()
    rank = scores.argsort(dim=-1, descending=True)
    target_positions = (rank == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    mean_rank = (target_positions.float() + 1.0).mean().item()
    mrr = (1.0 / (target_positions.float() + 1.0)).mean().item()
    return {"top1_acc": acc, "mean_rank": mean_rank, "mrr": mrr}
