from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocab: int = 10000,
        d: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=4 * d,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(ids) + self.pos(pos)
        key_padding_mask = ids == pad_id
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)
        # CLS = first token
        return x[:, 0, :]


class CrossEncoderTeacher(nn.Module):
    def __init__(
        self,
        *,
        vocab: int = 10000,
        d: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        d_img: int = 64,
        d_ctx: int = 16,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.img = nn.Linear(d_img, d)
        self.ctx = nn.Linear(d_ctx, d)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=4 * d,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, q: torch.Tensor, ad: torch.Tensor, img: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build a single packed sequence: [CLS] q [SEP] ad [SEP] + img_tok + ctx_tok
        B = q.shape[0]
        sep = torch.full((B, 1), 2, device=q.device, dtype=torch.long)
        ids = torch.cat([q, sep, ad, sep], dim=1)

        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(ids) + self.pos(pos)

        # append image/context tokens
        img_tok = self.img(img).unsqueeze(1)
        ctx_tok = self.ctx(ctx).unsqueeze(1)
        x = torch.cat([self.cls.expand(B, 1, -1), x, img_tok, ctx_tok], dim=1)

        key_padding_mask = torch.cat([
            torch.zeros((B, 1), device=q.device, dtype=torch.bool),
            ids == 0,
            torch.zeros((B, 2), device=q.device, dtype=torch.bool),
        ], dim=1)

        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=key_padding_mask)

        x = self.ln(x)
        rep = x[:, 0, :]
        logit = self.head(rep).squeeze(-1)
        return logit, rep


class StudentTwoTower(nn.Module):
    def __init__(
        self,
        *,
        vocab: int = 10000,
        d: int = 192,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        d_img: int = 64,
        d_ctx: int = 16,
        teacher_d: int = 384,
    ) -> None:
        super().__init__()
        self.q = TextBackbone(vocab=vocab, d=d, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.ad = TextBackbone(vocab=vocab, d=d, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.img = nn.Linear(d_img, d)
        self.ctx = nn.Linear(d_ctx, d)

        self.fuse = nn.Sequential(
            nn.Linear(d * 4, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.GELU(),
        )

        self.head = nn.Linear(d, 1)
        self.rep_proj = nn.Linear(d, teacher_d)

    def forward(self, q: torch.Tensor, ad: torch.Tensor, img: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_rep = self.q(q)
        ad_rep = self.ad(ad)
        img_rep = self.img(img)
        ctx_rep = self.ctx(ctx)

        fused = self.fuse(torch.cat([q_rep, ad_rep, img_rep, ctx_rep], dim=-1))
        logit = self.head(fused).squeeze(-1)
        return logit, self.rep_proj(fused)


def kd_loss(
    *,
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    y: torch.Tensor,
    student_rep: torch.Tensor,
    teacher_rep: torch.Tensor,
    alpha: float = 0.6,
    temp: float = 2.0,
    feat_weight: float = 0.1,
) -> torch.Tensor:
    hard = F.binary_cross_entropy_with_logits(student_logit, y)

    # logits KL (binary): treat as 2-class distribution
    s = torch.stack([torch.zeros_like(student_logit), student_logit / temp], dim=-1)
    t = torch.stack([torch.zeros_like(teacher_logit), teacher_logit / temp], dim=-1)
    soft = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t.detach(), dim=-1), reduction="batchmean") * (temp * temp)

    feat = F.mse_loss(student_rep, teacher_rep.detach())

    return alpha * hard + (1 - alpha) * soft + feat_weight * feat
