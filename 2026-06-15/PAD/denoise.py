from __future__ import annotations

import numpy as np


def build_popularity_gate(item_pop: np.ndarray, eta: float) -> np.ndarray:
    # s_i(eta) = (pop(i) / max_pop)^eta
    max_pop = float(item_pop.max()) if item_pop.size else 1.0
    if max_pop <= 0:
        max_pop = 1.0
    s = np.power(item_pop / max_pop, eta, dtype=np.float64)
    return np.clip(s, 0.0, 1.0)


def pad_weight(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    gate_s: np.ndarray,
    alpha: float,
    item_ids: np.ndarray,
) -> np.ndarray:
    """PAD weighting for BCE samples.

    Base denoising (RCE-style, Eq.14):
      w(u,i)=y_hat^alpha for y=1; w(u,i)=(1-y_hat)^alpha for y=0

    PAD gate (Eq.13):
      w_PAD=(1-s_i)+s_i*w

    gate_s is per-item s_i.
    """
    y_hat = np.clip(y_hat, 1e-6, 1 - 1e-6)
    pos_w = np.power(y_hat, alpha)
    neg_w = np.power(1.0 - y_hat, alpha)
    base = np.where(y_true > 0.5, pos_w, neg_w)

    s = gate_s[item_ids]
    return (1.0 - s) + s * base
