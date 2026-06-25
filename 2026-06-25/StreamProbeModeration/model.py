from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ToyLMConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    seq_len: int = 64


class ToyBlock(nn.Module):
    """A position-wise residual block (no attention).

    We intentionally avoid token mixing so that token-level safety signals remain
    readable throughout layers, making this a stable toy setup for hidden-state
    probes.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.ln(x))


class ToyCausalTransformer(nn.Module):
    """A tiny frozen 'generator' that exposes per-layer hidden states.

    NOTE: Despite the name, this is a minimal token-wise stack (no attention),
    designed to keep the reproduction lightweight and deterministic.
    """

    def __init__(self, cfg: ToyLMConfig, trigger_tokens: tuple[int, ...] = (13, 37, 101)):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([ToyBlock(cfg.d_model) for _ in range(cfg.n_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Inject a toy 'safety feature' into embeddings so that unsafe triggers are
        # separable from hidden states (mimicking the premise that safety signals
        # exist in hidden activations).
        with torch.no_grad():
            # Make dim-0 a clean, separable "safety feature":
            # - safe tokens: negative
            # - trigger tokens: strong positive
            self.embed.weight[:, 0] = -1.0
            for t in trigger_tokens:
                if 0 <= t < cfg.vocab_size:
                    self.embed.weight[t, 0] = 6.0

    def forward(self, x: torch.Tensor, return_hiddens: bool = True):
        """x: (B, T) -> logits: (B, T, V)

        If return_hiddens, returns a list of hidden states per layer (including
        embedding output at index 0).
        """
        _, t = x.shape
        h = self.embed(x) + self.pos[:, :t]
        hiddens = [h]
        for blk in self.blocks:
            h = blk(h)
            hiddens.append(h)
        logits = self.lm_head(h)
        if return_hiddens:
            return logits, hiddens
        return logits


class HiddenStateProbe(nn.Module):
    """Lightweight probe predicting streaming unsafe probability per token."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> probs: (B, T)"""
        return torch.sigmoid(self.net(h).squeeze(-1))
