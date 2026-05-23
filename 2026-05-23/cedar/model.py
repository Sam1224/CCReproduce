from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CedarOutput:
    z_centered: torch.Tensor
    z_tilde: torch.Tensor
    z_sparse: torch.Tensor
    z_hat: torch.Tensor
    active_mask: torch.Tensor


def topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    """Binary mask that keeps top-k abs coordinates per row."""
    if k <= 0 or k > x.shape[-1]:
        raise ValueError(f"k must be in [1, D], got k={k}, D={x.shape[-1]}")

    _, idx = torch.topk(x.abs(), k=k, dim=-1, largest=True, sorted=False)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(dim=-1, index=idx, value=True)
    return mask


class CEDAR(nn.Module):
    """CEDAR: orthogonal change-of-basis + top-k sparsity + reconstruction.

    Paper idea: learn an invertible transform U, apply a top-k bottleneck in that
    basis, then invert back. Here we use an orthogonal matrix U so inversion is
    simply transpose.

    z_tilde = U (z - b)
    z_sparse = topk(z_tilde)
    z_hat = U^T z_sparse + b
    """

    def __init__(self, dim: int, k: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.k = int(k)

        # Learn an orthogonal matrix via parametrization.
        # Initialized as identity for stable training.
        self._linear = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.eye_(self._linear.weight)
        self._linear = nn.utils.parametrizations.orthogonal(self._linear)

        self.register_buffer("mean", torch.zeros(self.dim))
        self._mean_initialized = False

    @torch.no_grad()
    def init_mean(self, z_batch: torch.Tensor) -> None:
        self.mean.copy_(z_batch.mean(dim=0))
        self._mean_initialized = True

    def forward(self, z: torch.Tensor) -> CedarOutput:
        if not self._mean_initialized:
            raise RuntimeError("Call init_mean() with a batch of embeddings before training.")

        z_centered = z - self.mean
        z_tilde = self._linear(z_centered)

        mask = topk_mask(z_tilde, self.k)
        z_sparse = z_tilde.masked_fill(~mask, 0.0)

        # Inverse is transpose for an orthogonal matrix.
        # nn.utils.parametrizations.orthogonal keeps weight orthogonal.
        weight = self._linear.weight
        z_hat = F.linear(z_sparse, weight.t()) + self.mean

        return CedarOutput(
            z_centered=z_centered,
            z_tilde=z_tilde,
            z_sparse=z_sparse,
            z_hat=z_hat,
            active_mask=mask,
        )


def reconstruction_loss(z_hat: torch.Tensor, z: torch.Tensor, kind: str = "l1") -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(z_hat, z)
    if kind == "l2":
        return F.mse_loss(z_hat, z)
    raise ValueError(f"Unknown loss kind: {kind}")
