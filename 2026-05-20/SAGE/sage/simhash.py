from __future__ import annotations

import torch


class SimHash:
    """A small SimHash implementation for stratifying behavioral vectors.

    We follow the paper's idea of using SimHash as a scalable locality-sensitive hashing
    primitive to bucket similar user behavior.
    """

    def __init__(self, n_bits: int, n_features: int, seed: int = 0):
        if n_bits <= 0 or n_bits > 63:
            raise ValueError("n_bits must be in [1, 63]")
        self.n_bits = n_bits
        self.n_features = n_features
        generator = torch.Generator().manual_seed(seed)
        self.proj = torch.randn((n_bits, n_features), generator=generator)

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        """Compute packed int64 SimHash codes.

        Args:
            x: [N, D] float tensor.

        Returns:
            codes: [N] int64 tensor.
        """
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"x must be [N, {self.n_features}]")

        bits = (x @ self.proj.T) > 0  # [N, n_bits]
        codes = torch.zeros((x.shape[0],), dtype=torch.int64, device=x.device)
        for bit_idx in range(self.n_bits):
            codes |= (bits[:, bit_idx].to(torch.int64) << bit_idx)
        return codes
