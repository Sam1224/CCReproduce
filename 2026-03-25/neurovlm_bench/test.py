from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import TASKS, make_dataset
from model import NeuroVLMProbe


def main() -> None:
    ckpt = torch.load("checkpoints/neurovlm.pt", map_location="cpu")
    model = NeuroVLMProbe(img_dim=int(ckpt["img_dim"]), task_vocab=len(TASKS), out_dim=4)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    test = make_dataset(n=2000, seed=2)

    with torch.no_grad():
        pred = model(test.img, test.task).argmax(dim=-1)

    overall = (pred == test.y).float().mean().item()
    print(f"overall_acc={overall:.3f}")

    for tid, name in enumerate(TASKS):
        m = (test.task == tid)
        if int(m.sum().item()) == 0:
            continue
        acc = (pred[m] == test.y[m]).float().mean().item()
        print(f"{name}_acc={acc:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
