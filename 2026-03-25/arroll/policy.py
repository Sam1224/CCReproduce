from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    hidden: int = 64


class DigitPolicy(nn.Module):
    """A tiny autoregressive policy over digits 0..9.

    Conditioned on the target (scalar), and previous digit tokens.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(10, cfg.hidden)
        self.target_emb = nn.Linear(1, cfg.hidden)
        self.rnn = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.head = nn.Linear(cfg.hidden, 10)

    def forward(self, target: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits for next token.

        target: (B, 1) float
        tokens: (B, T) int64
        """
        t = self.target_emb(target).unsqueeze(1)  # (B,1,H)
        x = self.token_emb(tokens)  # (B,T,H)
        x = x + t
        h, _ = self.rnn(x)
        return self.head(h[:, -1, :])  # (B,10)

    @torch.no_grad()
    def sample_next(self, target: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.forward(target, tokens)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
