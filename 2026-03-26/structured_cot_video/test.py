from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset
from model import CoTVideoModel, onehot


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("checkpoints/cot_video.pt", map_location=device)
    model = CoTVideoModel().to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    test = make_dataset(n=2000, seed=2)

    with torch.no_grad():
        step = model.step_logits(test.video.to(device)).argmax(dim=-1)
        ans = model(test.video.to(device), onehot(step, 5)).argmax(dim=-1).cpu()

    acc = (ans == test.y).float().mean().item()
    print(f"accuracy={acc:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
