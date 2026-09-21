from __future__ import annotations

from dataclasses import dataclass
from math import inf

import torch
from torch import nn


@dataclass
class Segment:
    start: int
    end: int
    coverage: float
    error_rate: float


class ConfidenceGate(nn.Module):
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def wilson_upper_bound(errors: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    phat = errors / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    radius = z * ((phat * (1 - phat) + z * z / (4 * n)) / n) ** 0.5
    return (center + radius) / denom


def certificate_coverage(confidence: torch.Tensor, correct: torch.Tensor, threshold: float, target_precision: float) -> tuple[float, float]:
    accepted = confidence >= threshold
    n = int(accepted.sum().item())
    if n == 0:
        return 0.0, 1.0
    errors = int((~correct[accepted]).sum().item())
    upper_error = wilson_upper_bound(errors, n)
    coverage = n / int(confidence.numel())
    return (coverage if 1.0 - upper_error >= target_precision else 0.0), upper_error


def best_contiguous_partition(group_quality: list[float], k: int) -> list[Segment]:
    n = len(group_quality)
    prefix = [0.0]
    for value in group_quality:
        prefix.append(prefix[-1] + value)
    dp = [[-inf] * (k + 1) for _ in range(n + 1)]
    prev = [[-1] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for parts in range(1, k + 1):
            for j in range(parts - 1, i):
                score = dp[j][parts - 1] + (prefix[i] - prefix[j]) / (i - j)
                if score > dp[i][parts]:
                    dp[i][parts] = score
                    prev[i][parts] = j
    segments = []
    i, parts = n, k
    while parts > 0:
        j = prev[i][parts]
        mean_quality = (prefix[i] - prefix[j]) / (i - j)
        segments.append(Segment(j, i - 1, mean_quality, 1.0 - mean_quality))
        i, parts = j, parts - 1
    return list(reversed(segments))
