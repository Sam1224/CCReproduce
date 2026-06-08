from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SAEConfig:
    embed_dim: int
    latent_dim: int = 256
    topk: int = 16


class IdentitySparseAutoencoder(nn.Module):
    """A minimal "SAE" that keeps the embedding space unchanged.

    For the toy reproduction we often work directly in the embedding basis
    (e.g. one-hot concept basis). In that case, a learned SAE is unnecessary.

    This module still exposes the same encode/decode API as a regular SAE, so
    the TEVI editor logic stays identical.
    """

    def __init__(self, embed_dim: int, topk: int = 0):
        super().__init__()
        self.cfg = SAEConfig(embed_dim=embed_dim, latent_dim=embed_dim, topk=topk)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        if self.cfg.topk <= 0 or self.cfg.topk >= z.shape[-1]:
            return z
        topk_vals, topk_idx = torch.topk(z, k=self.cfg.topk, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class TopKSparseAutoencoder(nn.Module):
    """A simple TopK sparse autoencoder.

    This is a lightweight stand-in for the SAE family used in TEVI.
    We implement: z = relu(W_e x + b_e); keep only top-k entries of z.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.embed_dim, cfg.latent_dim)
        self.decoder = nn.Linear(cfg.latent_dim, cfg.embed_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x))
        if self.cfg.topk <= 0 or self.cfg.topk >= z.shape[-1]:
            return z
        topk_vals, topk_idx = torch.topk(z, k=self.cfg.topk, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class TEVIMasker(nn.Module):
    """Caption-conditioned masking module.

    Given text embedding t, predict mask m in [0,1]^latent_dim.
    """

    def __init__(self, text_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(t))


class TEVI(nn.Module):
    """TEVI-style embedding editor: x_edit = Decoder(z ⊙ mask(text))."""

    def __init__(self, sae: nn.Module, masker: TEVIMasker):
        super().__init__()
        self.sae = sae
        self.masker = masker

    def edit(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        z = self.sae.encode(image_emb)
        m = self.masker(text_emb)
        x_edit = self.sae.decode(z * m)
        return F.normalize(x_edit, dim=-1)


def info_nce_loss(sim: torch.Tensor) -> torch.Tensor:
    """Standard symmetric InfoNCE for NxN similarity matrix."""
    n = sim.shape[0]
    targets = torch.arange(n, device=sim.device)
    loss_i2t = F.cross_entropy(sim, targets)
    loss_t2i = F.cross_entropy(sim.T, targets)
    return 0.5 * (loss_i2t + loss_t2i)
