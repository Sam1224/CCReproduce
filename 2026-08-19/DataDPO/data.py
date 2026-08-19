from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DataBundle:
    pool_x: torch.Tensor  # [N, d]
    pool_y: torch.Tensor  # [N]
    pool_quality: torch.Tensor  # [N]

    val_x: torch.Tensor  # [V, d]
    val_y: torch.Tensor  # [V]

    test_x: torch.Tensor  # [T, d]
    test_y: torch.Tensor  # [T]


def _make_linear_world(seed: int, d: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, generator=g)
    return w / (w.norm() + 1e-8)


def _sample(
    *,
    w: torch.Tensor,
    n: int,
    noise: float,
    g: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = w.numel()
    x = torch.randn(n, d, generator=g)
    logits = x @ w + torch.randn(n, generator=g) * noise
    y = (logits > 0).long()
    return x, y


def build_data(
    *,
    seed: int = 13,
    d: int = 24,
    pool_size: int = 900,
    val_size: int = 400,
    test_size: int = 900,
) -> DataBundle:
    g = torch.Generator().manual_seed(seed)

    w_in = _make_linear_world(seed + 1, d)

    # Out-of-domain direction is intentionally *misaligned* with the in-domain
    # decision boundary, making OOD samples harmful if included during training.
    w_out = -w_in

    # Candidate pool: mixture
    n_easy = int(pool_size * 0.30)
    n_hard = int(pool_size * 0.30)
    n_ood = pool_size - n_easy - n_hard

    x_easy, y_easy = _sample(w=w_in, n=n_easy, noise=0.15, g=g)
    x_hard, y_hard = _sample(w=w_in, n=n_hard, noise=0.55, g=g)
    x_ood, y_ood = _sample(w=w_out, n=n_ood, noise=0.25, g=g)

    pool_x = torch.cat([x_easy, x_hard, x_ood], dim=0)
    pool_y = torch.cat([y_easy, y_hard, y_ood], dim=0)

    # External quality: in-domain=1, out-of-domain=0
    pool_quality = torch.cat(
        [torch.ones(n_easy + n_hard), torch.zeros(n_ood)], dim=0
    )

    # Shuffle pool
    perm = torch.randperm(pool_size, generator=g)
    pool_x = pool_x[perm]
    pool_y = pool_y[perm]
    pool_quality = pool_quality[perm]

    # Clean validation/test: in-domain
    val_x, val_y = _sample(w=w_in, n=val_size, noise=0.20, g=g)
    test_x, test_y = _sample(w=w_in, n=test_size, noise=0.20, g=g)

    # Normalize for stability
    pool_x = pool_x / (pool_x.norm(dim=-1, keepdim=True) + 1e-8)
    val_x = val_x / (val_x.norm(dim=-1, keepdim=True) + 1e-8)
    test_x = test_x / (test_x.norm(dim=-1, keepdim=True) + 1e-8)

    return DataBundle(
        pool_x=pool_x,
        pool_y=pool_y,
        pool_quality=pool_quality,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def softmax_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, y)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a @ b.T
