from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class QuantizeOutput:
    quantized: torch.Tensor  # (B, D)
    codes: torch.Tensor  # (B, L) long
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


class ResidualVectorQuantizer(nn.Module):
    """Residual Quantization (RQ) as a lightweight RQ-VAE-style component.

    Given x, we sequentially quantize residuals with multiple codebooks.

    This is a toy implementation; the goal is to produce *hierarchical discrete codes*
    resembling the paper's SIDs.
    """

    def __init__(self, dim: int, num_codebooks: int = 3, codebook_size: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> QuantizeOutput:
        # x: (B, D)
        residual = x
        quantized_sum = torch.zeros_like(x)

        codes = []
        codebook_loss = torch.zeros((), device=x.device)
        commitment_loss = torch.zeros((), device=x.device)

        for level in range(self.num_codebooks):
            cb = self.codebooks[level]  # (K, D)

            # (B, K): squared L2 distance
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ cb.t()
                + cb.pow(2).sum(dim=1, keepdim=True).t()
            )
            idx = torch.argmin(dist, dim=1)  # (B,)
            codes.append(idx)

            e = cb[idx]  # (B, D)

            # VQ-VAE losses
            codebook_loss = codebook_loss + torch.mean((e - residual.detach()) ** 2)
            commitment_loss = commitment_loss + torch.mean((e.detach() - residual) ** 2)

            # straight-through
            e_st = residual + (e - residual).detach()

            quantized_sum = quantized_sum + e_st
            residual = residual - e_st

        return QuantizeOutput(
            quantized=quantized_sum,
            codes=torch.stack(codes, dim=1),
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )


def info_nce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """In-batch InfoNCE for (q, d) positives with other docs as negatives."""

    logits = (q @ d.t()) / temperature  # (B, B)
    labels = torch.arange(q.shape[0], device=q.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def rq_contrastive_loss(
    q: torch.Tensor,
    d_pre: torch.Tensor,
    d_quant: torch.Tensor,
    codebook_loss: torch.Tensor,
    commitment_loss: torch.Tensor,
    beta: float = 0.25,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Query-bridged contrastive quantization loss.

    - Use query embeddings as supervision to shape quantization partitions.
    - Combine InfoNCE(q, quantized_item) + VQ losses.

    d_pre is kept for potential extensions (e.g., reconstruction), but unused here.
    """

    _ = d_pre
    nce = info_nce_loss(q, d_quant, temperature=temperature)
    vq = codebook_loss + beta * commitment_loss
    return nce + vq
