from __future__ import annotations

# Smoke test: import modules and run a tiny forward.

import torch

from model import MFConfig, init_params, predict


def test_forward() -> None:
    cfg = MFConfig(n_users=10, n_items=20, dim=8)
    params = init_params(cfg, seed=0)
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([3, 4, 5])
    out = predict(params, u, i)
    assert out.shape == (3,)


if __name__ == "__main__":
    test_forward()
    print("ok")
