from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import VideoReasoner


def main() -> None:
    ckpt = torch.load("checkpoints/perceptioncomp.pt", map_location="cpu")
    model = VideoReasoner()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    test = make_dataset(n=2000, seed=2)
    with torch.no_grad():
        pred = model(test.video).argmax(dim=-1)
    acc = (pred == test.y).float().mean().item()
    print(f"accuracy={acc:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
