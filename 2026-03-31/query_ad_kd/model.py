from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab: int, d: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids).mean(dim=0)


class Teacher(nn.Module):
    def __init__(self, vocab: int = 5000, d_img: int = 64, d: int = 256) -> None:
        super().__init__()
        self.q = TextEncoder(vocab, d)
        self.t = TextEncoder(vocab, d)
        self.img = nn.Linear(d_img, d)
        self.mlp = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, q: torch.Tensor, title: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.q(q), self.t(title), self.img(img)], dim=-1)
        return self.mlp(feat).squeeze(-1)


class StudentTwoTower(nn.Module):
    def __init__(self, vocab: int = 5000, d_img: int = 64, d: int = 96) -> None:
        super().__init__()
        self.q = TextEncoder(vocab, d)
        self.title = TextEncoder(vocab, d)
        self.img = nn.Linear(d_img, d)
        self.head = nn.Linear(d * 2, 1)

    def forward(self, q: torch.Tensor, title: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        q_emb = self.q(q)
        ad_emb = (self.title(title) + self.img(img)) / 2
        return self.head(torch.cat([q_emb, ad_emb], dim=-1)).squeeze(-1)


def distill_loss(student_logit: torch.Tensor, teacher_logit: torch.Tensor, y: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    # alpha * hard labels + (1-alpha) * soft targets
    hard = F.binary_cross_entropy_with_logits(student_logit, y)
    soft = F.mse_loss(torch.sigmoid(student_logit), torch.sigmoid(teacher_logit.detach()))
    return alpha * hard + (1 - alpha) * soft
