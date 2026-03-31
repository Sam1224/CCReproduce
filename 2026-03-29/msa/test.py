from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import MemorySparseAttention


def main() -> None:
    data = make_dataset(n=200, seed=42)
    model = MemorySparseAttention(d=64, heads=4, topk=8, num_classes=12)

    ex = data[0]
    logits = model(ex.docs, ex.query)
    assert logits.shape == (12,)

    ckpt = Path("checkpoints/msa.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        logits2 = model(ex.docs, ex.query)
        assert logits2.shape == (12,)

    print("msa smoke test ok")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
