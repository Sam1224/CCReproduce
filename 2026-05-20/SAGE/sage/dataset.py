from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ToyPUData:
    x: torch.Tensor  # [N, D]
    y_true: torch.Tensor  # [N] {0,1}
    labeled_positive: torch.Tensor  # [N] bool
    cohort_id: torch.Tensor  # [N] int64
    edge_cohort: torch.Tensor  # [N] bool


def make_toy_pu_dataset(
    *,
    n_legit: int = 18000,
    n_fraud: int = 2000,
    n_features: int = 16,
    n_legit_cohorts: int = 6,
    labeled_pos_fraction: float = 0.2,
    seed: int = 42,
) -> ToyPUData:
    """Create a toy PU dataset with an explicit 'edge cohort'.

    Intuition aligned with the paper:
    - Fraud (positive) is rare and only partially labeled.
    - Legit users come from multiple cohorts.
    - One cohort is an 'edge case' whose behavior distribution overlaps with fraud,
      causing typical fraud-first models to generate false positives.

    Returns:
        ToyPUData with full ground truth for evaluation.
    """

    rng = np.random.default_rng(seed)

    # Cohort means.
    legit_means = rng.normal(0.0, 2.0, size=(n_legit_cohorts, n_features)).astype(np.float32)

    # Fraud mean is close to cohort 0 to emulate 'super-fans / sleep-music sessions'.
    fraud_mean = legit_means[0] + rng.normal(0.0, 0.5, size=(n_features,)).astype(np.float32)

    # Legit samples.
    # Cohort distribution is intentionally long-tailed: cohort 0 is rare but overlaps with fraud,
    # emulating edge cases that are under-represented by naive sampling.
    if n_legit_cohorts < 2:
        raise ValueError("n_legit_cohorts must be >= 2")
    weights = np.array([0.02] + [0.98 / (n_legit_cohorts - 1)] * (n_legit_cohorts - 1), dtype=np.float64)
    legit_cohort_ids = rng.choice(np.arange(n_legit_cohorts), size=(n_legit,), p=weights)
    legit_x = legit_means[legit_cohort_ids] + rng.normal(0.0, 1.0, size=(n_legit, n_features)).astype(
        np.float32
    )

    # Fraud samples: mixture near fraud_mean and another distant component.
    fraud_component = rng.integers(0, 2, size=(n_fraud,))
    fraud_means = np.stack(
        [
            fraud_mean,
            fraud_mean + rng.normal(0.0, 3.0, size=(n_features,)).astype(np.float32),
        ],
        axis=0,
    )
    fraud_x = fraud_means[fraud_component] + rng.normal(0.0, 1.0, size=(n_fraud, n_features)).astype(
        np.float32
    )

    x = np.concatenate([legit_x, fraud_x], axis=0)
    y_true = np.concatenate([np.zeros(n_legit, dtype=np.int64), np.ones(n_fraud, dtype=np.int64)], axis=0)
    cohort_id = np.concatenate([legit_cohort_ids, rng.integers(0, n_legit_cohorts, size=(n_fraud,))])

    edge_cohort = cohort_id == 0

    # Only a fraction of fraud is labeled positive.
    labeled_positive = np.zeros((n_legit + n_fraud,), dtype=bool)
    fraud_indices = np.arange(n_legit, n_legit + n_fraud)
    n_labeled = int(round(n_fraud * labeled_pos_fraction))
    labeled_idx = rng.choice(fraud_indices, size=n_labeled, replace=False)
    labeled_positive[labeled_idx] = True

    return ToyPUData(
        x=torch.from_numpy(x),
        y_true=torch.from_numpy(y_true),
        labeled_positive=torch.from_numpy(labeled_positive),
        cohort_id=torch.from_numpy(cohort_id.astype(np.int64)),
        edge_cohort=torch.from_numpy(edge_cohort),
    )
