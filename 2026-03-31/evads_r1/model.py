from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEncoder(nn.Module):
    def __init__(self, vocab: int, d: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        return x.mean(dim=0)


class EVAdsPolicy(nn.Module):
    def __init__(self, vocab: int = 1000, d_v: int = 48, d: int = 128, n_classes: int = 12) -> None:
        super().__init__()
        self.asr = TinyEncoder(vocab, d)
        self.ocr = TinyEncoder(vocab, d)
        self.q = TinyEncoder(vocab, d)
        self.v = nn.Linear(d_v, d)

        self.head = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.GELU(),
            nn.Linear(d, n_classes),
        )

    def forward(self, video: torch.Tensor, asr: torch.Tensor, ocr: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        v = self.v(video).mean(dim=0)
        a = self.asr(asr)
        o = self.ocr(ocr)
        qq = self.q(q)
        return self.head(torch.cat([v, a, o, qq], dim=-1))


def mg_grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    # group-relative advantage (z-score)
    mu = rewards.mean()
    std = rewards.std().clamp_min(1e-6)
    return (rewards - mu) / std


def multinomial_logprob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(1, actions.unsqueeze(1)).squeeze(1)
