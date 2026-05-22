import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _gaussian_blur_like(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    # Simple separable blur (no external deps)
    device = x.device
    t = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(t**2) / (2 * (k / 6) ** 2))
    g = g / g.sum()
    g2d = (g[:, None] * g[None, :]).view(1, 1, k, k)
    x = torch.nn.functional.conv2d(x, g2d, padding=k // 2)
    return x


def compute_radial_spectrum(img: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Compute 1D radial log-power spectrum (toy implementation).

    img: (1, H, W) in [0, 1]
    returns: (bins,) float tensor
    """
    assert img.ndim == 3 and img.shape[0] == 1
    h, w = img.shape[-2:]

    x = img.squeeze(0)
    f = torch.fft.fft2(x)
    f = torch.fft.fftshift(f)
    power = (f.real**2 + f.imag**2).clamp_min(1e-12)
    logp = torch.log(power)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device),
        torch.arange(w, device=img.device),
        indexing="ij",
    )
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max().clamp_min(1.0)
    rbin = torch.clamp((rr / rmax * (bins - 1)).long(), 0, bins - 1)

    spec = torch.zeros(bins, device=img.device)
    cnt = torch.zeros(bins, device=img.device)
    spec.scatter_add_(0, rbin.view(-1), logp.view(-1))
    cnt.scatter_add_(0, rbin.view(-1), torch.ones_like(logp).view(-1))
    spec = spec / cnt.clamp_min(1.0)
    return spec


def tail_uplift_score(spec: torch.Tensor, tail_ratio: float = 0.15) -> torch.Tensor:
    """A scalar score: average tail residual over a fitted power-law slope."""
    n = spec.numel()
    idx = torch.arange(n, device=spec.device, dtype=torch.float32) + 1
    x = torch.log(idx)
    y = spec

    # Fit y = a + b*x via least squares
    x_mean = x.mean()
    y_mean = y.mean()
    b = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum().clamp_min(1e-6)
    a = y_mean - b * x_mean
    y_hat = a + b * x

    t0 = int((1.0 - tail_ratio) * n)
    residual = (y[t0:] - y_hat[t0:]).mean()
    return residual


class ToyAIGenDataset(Dataset):
    def __init__(self, n: int = 2000, image_size: int = 64, seed: int = 0):
        self.n = n
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        h = w = self.image_size
        base = torch.tensor(self.rng.standard_normal((1, 1, h, w)), dtype=torch.float32)
        base = base / base.std().clamp_min(1e-6)
        base = _gaussian_blur_like(base, k=9)
        base = (base - base.min()) / (base.max() - base.min() + 1e-6)

        label = int(self.rng.random() < 0.5)
        x = base.clone()

        if label == 1:
            # "fake": inject ultra-high-frequency energy
            hf = torch.tensor(self.rng.standard_normal((1, 1, h, w)), dtype=torch.float32)
            hf = hf - _gaussian_blur_like(hf, k=5)
            x = (x + 0.20 * hf).clamp(0.0, 1.0)

        return x.squeeze(0), label
