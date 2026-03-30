from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgEncoder(nn.Module):
    def __init__(self, d_in: int = 32, d_out: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TxtEncoder(nn.Module):
    def __init__(self, vocab: int = 256, emb: int = 64, d_out: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.proj = nn.Linear(emb, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x).mean(dim=1)
        return F.normalize(self.proj(e), dim=-1)


def contrastive(img_emb: torch.Tensor, txt_emb: torch.Tensor, t: float = 0.07) -> torch.Tensor:
    logits = img_emb @ txt_emb.t() / t
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
