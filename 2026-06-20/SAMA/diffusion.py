"""Anchor-Preserving Diffusion for image synthesis (SAMA Sec III.E).

A compact DDPM operating directly on the toy [3,8,8] image as the "latent" z0
(a real Stable-Diffusion VAE is omitted offline; see note below). Two paper
enhancements are implemented:
  * Anchor-Weighted Prompt (Eq. 10): P = "A photo of (A_entity : w) ..." — entity
    anchor tokens get emphasis weight w > 1 in the conditioning embedding.
  * Spatially-Constrained Denoising / masked latent blending (Eq. 11):
        z_{t-1} = M ⊙ z^src_{t-1} + (1 - M) ⊙ z^gen_{t-1}
    anchor (foreground) pixels are kept from the forward-noised original (identity
    preserved); background is synthesised from the prompt (diversity).

NOTE: paper uses an LDM (VAE latents). Here z0 == image for a self-contained run;
the masking/blending and conditioning logic is identical and would transfer to a
VAE latent unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import IMG_C, IMG_H, IMG_W, STOI, VOCAB, encode

V = len(VOCAB)
DC = 32          # diffusion conditioning dim
T = 20           # diffusion steps (toy)


def make_schedule(t_steps=T):
    betas = torch.linspace(1e-4, 0.2, t_steps)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, 0)
    return betas, alphas, abar


BETAS, ALPHAS, ABAR = make_schedule()


class PromptEncoder(nn.Module):
    """Anchor-weighted prompt embedding (Eq. 10)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, DC, padding_idx=STOI["<pad>"])

    def forward(self, ids, weights):          # ids [B,L], weights [B,L]
        e = self.emb(ids)                     # [B,L,DC]
        w = weights.unsqueeze(-1)             # emphasis on entity-anchor tokens
        return (e * w).sum(1) / (w.sum(1) + 1e-6)   # [B,DC] weighted prompt


class TinyUNet(nn.Module):
    """Compact noise predictor eps_theta(z_t, t, prompt) with FiLM conditioning."""

    def __init__(self):
        super().__init__()
        self.t_emb = nn.Embedding(T, DC)
        self.in_conv = nn.Conv2d(IMG_C, 16, 3, padding=1)
        self.mid = nn.Conv2d(16, 16, 3, padding=1)
        self.out_conv = nn.Conv2d(16, IMG_C, 3, padding=1)
        self.film = nn.Linear(DC, 16 * 2)      # scale/shift from (t + prompt)
        self.cond_proj = nn.Linear(DC, DC)

    def forward(self, z, t, prompt):
        c = self.t_emb(t) + self.cond_proj(prompt)        # [B,DC]
        scale, shift = self.film(c).chunk(2, -1)          # FiLM params
        h = F.silu(self.in_conv(z))
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(self.mid(h))
        return self.out_conv(h)                            # predicted noise


def q_sample(z0, t, noise):
    """Forward diffusion: z_t = sqrt(abar_t) z0 + sqrt(1-abar_t) noise."""
    a = ABAR[t].view(-1, 1, 1, 1)
    return a.sqrt() * z0 + (1 - a).sqrt() * noise


def diffusion_loss(unet, z0, prompt):
    B = z0.size(0)
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(z0)
    zt = q_sample(z0, t, noise)
    pred = unet(zt, t, prompt)
    return F.mse_loss(pred, noise)


@torch.no_grad()
def sample(unet, z0_src, prompt, mask, strength=0.8):
    """Img2img reverse diffusion with anchor-preserving masked blending (Eq. 11).

    strength controls how much noise is added to the source (paper uses 0.8).
    """
    B = z0_src.size(0)
    t_start = int(T * strength) - 1
    noise = torch.randn_like(z0_src)
    z = q_sample(z0_src, torch.full((B,), t_start, dtype=torch.long), noise)
    for ti in range(t_start, -1, -1):
        t = torch.full((B,), ti, dtype=torch.long)
        eps = unet(z, t, prompt)
        a = ABAR[ti]
        z0_pred = (z - (1 - a).sqrt() * eps) / a.sqrt()
        z0_pred = z0_pred.clamp(0, 1)
        if ti > 0:
            ab_prev = ABAR[ti - 1]
            z_gen = ab_prev.sqrt() * z0_pred + (1 - ab_prev).sqrt() * torch.randn_like(z)
            # forward-noised original at t-1 (z^src)
            z_src = q_sample(z0_src, torch.full((B,), ti - 1, dtype=torch.long),
                             torch.randn_like(z))
            z = mask * z_src + (1 - mask) * z_gen          # Eq. 11
        else:
            z = mask * z0_src + (1 - mask) * z0_pred       # final blend
    return z.clamp(0, 1)
