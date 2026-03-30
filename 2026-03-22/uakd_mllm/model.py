from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, txt_vocab: int = 128, txt_dim: int = 32, img_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.txt_emb = nn.Embedding(txt_vocab, txt_dim)
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        t = self.txt_emb(txt).mean(dim=1)
        h = self.img_proj(img) + self.txt_proj(t)
        return self.head(h)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-9).log())).sum(dim=-1)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature**2)


def weighted_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, weights: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    # per-sample KL
    per = F.kl_div(s, t, reduction="none").sum(dim=-1) * (temperature**2)
    w = weights / (weights.mean().clamp_min(1e-6))
    return (per * w).mean()
