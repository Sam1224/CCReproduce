from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import GraphReranker


def mrr(scores: torch.Tensor, y: torch.Tensor) -> float:
    idx = torch.argsort(scores, descending=True)
    for rank, i in enumerate(idx.tolist(), start=1):
        if int(y[i].item()) == 1:
            return 1.0 / rank
    return 0.0


def main() -> None:
    ckpt = torch.load("checkpoints/grapher.pt", map_location="cpu")
    model = GraphReranker(d=64)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    data = make_dataset(n=1200, seed=2)
    _, te = split(data, 0.0)

    mrr_sum = 0.0
    with torch.no_grad():
        for ex in te[:300]:
            s = model(ex.q, ex.docs)
            mrr_sum += mrr(s, ex.y)

    print(f"MRR@{8}={mrr_sum/300:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
