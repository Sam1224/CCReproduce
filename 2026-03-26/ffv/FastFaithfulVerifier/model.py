from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class ScaledSinusoidalPositionalEmbedding(nn.Module):
    """Toy RoPE-scaling substitute.

    The paper discusses RoPE scaling for long-context verification.
    Implementing true RoPE requires attention-level changes; for a faithful *pipeline* we
    use a positional embedding with a controllable frequency scaling factor.

    For sequence length L and base_len, scale = max(1, L/base_len).
    """

    def __init__(self, d_model: int, base_len: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        self.base_len = base_len

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        scale = max(1.0, float(seq_len) / float(self.base_len))
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model)
        )
        div_term = div_term / scale
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


@dataclass
class VerifierOutput:
    span_logits: torch.Tensor  # (B,L)
    exit_logits_list: List[torch.Tensor]  # each (B,)
    exit_layer: torch.Tensor  # scalar int (0-index)


class OneModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        early_exit_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.early_exit_threshold = early_exit_threshold

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = ScaledSinusoidalPositionalEmbedding(d_model)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=False,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Adaptive early exit classifiers.
        self.exit_classifiers = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_layers)])

        # Token-level unsupported span detector.
        self.span_detector = nn.Linear(d_model, 1)

    def freeze_modules(self) -> None:
        # Nothing mandatory to freeze in this toy impl.
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_mean(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # h: (B,L,d) ; attn_mask: (B,L) bool
        mask = attn_mask.unsqueeze(-1).type_as(h)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (h * mask).sum(dim=1) / denom

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> VerifierOutput:
        input_ids = batch["input_ids"]  # (B,L)
        attn_mask = batch["attn_mask"]  # (B,L) bool

        bsz, seq_len = input_ids.shape
        x = self.embed(input_ids)  # (B,L,d)
        x = x + self.pos(seq_len, x.device).unsqueeze(0)

        # Transformer expects (L,B,d)
        h = x.transpose(0, 1)
        key_padding_mask = ~attn_mask  # True for pad

        exit_logits_list: List[torch.Tensor] = []
        exit_layer = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            h = layer(h, src_key_padding_mask=key_padding_mask)
            h_b = h.transpose(0, 1)  # (B,L,d)

            pooled = self._masked_mean(h_b, attn_mask)
            exit_logit = self.exit_classifiers[i](pooled).squeeze(-1)  # (B,)
            exit_logits_list.append(exit_logit)

            if not self.training:
                conf = torch.sigmoid(exit_logit)
                if torch.all(conf >= self.early_exit_threshold):
                    exit_layer = i
                    break

        h_b = h.transpose(0, 1)
        span_logits = self.span_detector(h_b).squeeze(-1)

        return VerifierOutput(span_logits=span_logits, exit_logits_list=exit_logits_list, exit_layer=torch.tensor(exit_layer))


def verifier_loss(
    out: VerifierOutput,
    span_labels: torch.Tensor,
    halluc: torch.Tensor,
    attn_mask: torch.Tensor,
    span_weight: float = 1.0,
    exit_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Span BCE on valid tokens.
    span_loss = F.binary_cross_entropy_with_logits(
        out.span_logits[attn_mask],
        span_labels[attn_mask],
    )

    # Early-exit supervision on each layer output.
    exit_losses = [F.binary_cross_entropy_with_logits(logit, halluc) for logit in out.exit_logits_list]
    exit_loss = torch.stack(exit_losses).mean() if exit_losses else torch.tensor(0.0, device=span_loss.device)

    total = span_weight * span_loss + exit_weight * exit_loss
    return total, {"span_loss": float(span_loss.detach()), "exit_loss": float(exit_loss.detach())}
