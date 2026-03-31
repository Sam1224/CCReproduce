from __future__ import annotations

import torch

from dataset import AwaResToyDataset, collate_sft
from model import AwaResConfig, AwaResModel


def test_shapes() -> None:
    ds = AwaResToyDataset(n_samples=8, seed=0)
    b = collate_sft([ds[i] for i in range(8)])

    model = AwaResModel(AwaResConfig())
    action_logits, ans_low, low_feat = model.forward_low(b.img_low)
    ans_fused = model.forward_with_crop(low_feat, b.img_crop)

    assert action_logits.shape == (8, 10)
    assert ans_low.shape == (8, 10)
    assert ans_fused.shape == (8, 10)


if __name__ == "__main__":
    test_shapes()
    print("ok")
