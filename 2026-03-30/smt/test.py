from __future__ import annotations

import torch

from model import SMTConfig, StandardModelTemplate


def test_forward() -> None:
    cfg = SMTConfig(n_cat=10, x_dim=4, cat_dim=8, x_proj=8)
    m = StandardModelTemplate(cfg)
    cat = torch.tensor([1, 2, 3])
    x = torch.randn(3, 4)
    logits = m(cat, x)
    assert logits.shape == (3,)


if __name__ == "__main__":
    test_forward()
    print("ok")
