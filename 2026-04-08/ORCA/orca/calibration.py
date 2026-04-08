from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
import torch

from .utils import EvalStats


@dataclass(frozen=True)
class StopResult:
    stop_steps: np.ndarray  # [N]
    correct: np.ndarray  # [N]


def first_crossing_stop(prob: np.ndarray, tau: float) -> np.ndarray:
    """Stop at the first step whose prob>=tau; fallback to last step."""
    n, t = prob.shape
    stop = np.full((n,), t - 1, dtype=np.int64)
    for i in range(n):
        idx = np.where(prob[i] >= tau)[0]
        if idx.size > 0:
            stop[i] = int(idx[0])
    return stop


def evaluate_policy(prob: np.ndarray, y: np.ndarray, tau: float) -> EvalStats:
    stop = first_crossing_stop(prob, tau)
    chosen = y[np.arange(y.shape[0]), stop]

    risk = float(1.0 - chosen.mean())
    avg_stop = float(stop.mean())
    savings = float(1.0 - (avg_stop + 1.0) / y.shape[1])
    return EvalStats(risk=risk, savings=savings, avg_stop_step=avg_stop)


def calibrate_tau(
    prob: np.ndarray,
    y: np.ndarray,
    delta: float,
    grid: Iterable[float] | None = None,
) -> Tuple[float, EvalStats]:
    """Select tau so that risk<=delta and savings is maximized.

    This is a simple grid-search calibration over tau in [0,1].
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    # A small safety margin to reduce calibration overfitting.
    target = max(0.0, delta - 0.02)

    best_tau = 1.0
    best_stats = evaluate_policy(prob, y, tau=1.0)

    for tau in grid:
        stats = evaluate_policy(prob, y, tau=float(tau))
        if stats.risk <= target and stats.savings >= best_stats.savings:
            best_tau = float(tau)
            best_stats = stats

    return best_tau, best_stats


@torch.no_grad()
def probe_probs(model, dataset, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for item in dataset:
        xs.append(item["x"].unsqueeze(0))
        ys.append(item["y"].unsqueeze(0))
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).cpu().numpy()
    logits = model(x).sigmoid().cpu().numpy()
    return logits, y
