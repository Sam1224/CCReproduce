from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    target: int
    length: int


def sample_episode(rng: np.random.Generator, length: int = 6, target_max: int = 30) -> Episode:
    return Episode(target=int(rng.integers(0, target_max + 1)), length=length)


def verifiable_reward(seq: list[int], target: int) -> float:
    return 1.0 if sum(seq) == target else 0.0


def feasible_prefix(prefix_sum: int, steps_done: int, length: int, target: int) -> bool:
    """Online feasibility check.

    With remaining steps, can we still reach target sum?
    Digits are in [0, 9].
    """
    remaining = length - steps_done
    min_possible = prefix_sum
    max_possible = prefix_sum + 9 * remaining
    return (min_possible <= target) and (target <= max_possible)
