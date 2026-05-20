from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MahalanobisGate:
    """Global statistical gate based on (shrunk) Mahalanobis distance."""

    shrinkage: float = 0.1
    quantile: float = 0.95
    eps: float = 1e-6

    mean: torch.Tensor | None = None
    inv_cov: torch.Tensor | None = None
    threshold: float | None = None

    @torch.no_grad()
    def fit(self, x_ref: torch.Tensor) -> "MahalanobisGate":
        x_ref = x_ref.float()
        self.mean = x_ref.mean(dim=0)
        centered = x_ref - self.mean
        cov = centered.T @ centered / max(1, x_ref.shape[0] - 1)
        diag = torch.diag(torch.diag(cov))
        cov = (1 - self.shrinkage) * cov + self.shrinkage * diag
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * self.eps
        self.inv_cov = torch.linalg.inv(cov)
        d2 = self.distance2(x_ref)
        self.threshold = torch.quantile(d2, self.quantile).item()
        return self

    @torch.no_grad()
    def distance2(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Gate not fitted")
        centered = x.float() - self.mean
        left = centered @ self.inv_cov
        return (left * centered).sum(dim=1)

    @torch.no_grad()
    def vote(self, x: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("Gate not fitted")
        return self.distance2(x) <= self.threshold


@dataclass
class KNNDensityGate:
    """Local density gate using kNN distances (brute-force, toy scale)."""

    k: int = 10
    quantile: float = 0.2
    eps: float = 1e-6

    x_ref: torch.Tensor | None = None
    threshold: float | None = None

    @torch.no_grad()
    def fit(self, x_ref: torch.Tensor) -> "KNNDensityGate":
        self.x_ref = x_ref.float()
        density = self.density(self.x_ref)
        self.threshold = torch.quantile(density, 1.0 - self.quantile).item()
        return self

    @torch.no_grad()
    def density(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_ref is None:
            raise RuntimeError("Gate not fitted")
        # [N, M] distances; for toy sizes only.
        dist = torch.cdist(x.float(), self.x_ref)
        k_eff = min(self.k + 1, dist.shape[1])
        nearest, _ = torch.topk(dist, k=k_eff, largest=False)
        # skip self-match when x is x_ref
        mean_dist = nearest[:, 1:].mean(dim=1)
        return 1.0 / (mean_dist + self.eps)

    @torch.no_grad()
    def vote(self, x: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("Gate not fitted")
        return self.density(x) >= self.threshold
