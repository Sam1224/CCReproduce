from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RQVAEOutput:
    z_e: torch.Tensor
    z_q: torch.Tensor
    code_indices: torch.Tensor  # [B, num_codebooks]
    recon: torch.Tensor
    loss_recon: torch.Tensor
    loss_commit: torch.Tensor


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, num_codebooks: int, codebook_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [B, D]
        residual = z
        all_indices = []
        quantized_sum = torch.zeros_like(z)

        for i in range(self.num_codebooks):
            codebook = self.codebooks[i]  # [K, D]
            # squared l2 distance: (z - e)^2 = z^2 + e^2 - 2 z e
            z2 = (residual**2).sum(dim=-1, keepdim=True)  # [B, 1]
            e2 = (codebook**2).sum(dim=-1).unsqueeze(0)  # [1, K]
            ze = residual @ codebook.t()  # [B, K]
            dist = z2 + e2 - 2 * ze
            indices = torch.argmin(dist, dim=-1)  # [B]
            all_indices.append(indices)
            e = F.embedding(indices, codebook)  # [B, D]
            quantized_sum = quantized_sum + e
            residual = residual - e

        code_indices = torch.stack(all_indices, dim=-1)  # [B, C]
        return quantized_sum, code_indices


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim, num_codebooks=num_codebooks, codebook_size=codebook_size
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, beta: float = 0.25) -> RQVAEOutput:
        z_e = self.encoder(x)
        z_q, code_indices = self.quantizer(z_e)

        # straight-through estimator
        z_st = z_e + (z_q - z_e).detach()
        recon = self.decoder(z_st)

        loss_recon = F.mse_loss(recon, x)
        loss_commit = beta * F.mse_loss(z_e, z_q.detach())
        return RQVAEOutput(
            z_e=z_e,
            z_q=z_q,
            code_indices=code_indices,
            recon=recon,
            loss_recon=loss_recon,
            loss_commit=loss_commit,
        )
