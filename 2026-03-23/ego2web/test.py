from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import EgoEncoder


def main() -> None:
    batch = make_dataset(n=1200, seed=2)
    ckpt = torch.load("checkpoints/ego2web.pt", map_location="cpu")

    model = EgoEncoder(vocab=64, num_tasks=8)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(batch.seq)
    acc = (logits.argmax(dim=-1) == batch.y).float().mean().item()
    print(f"acc={acc:.3f} (toy Ego2Web task classification)")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
