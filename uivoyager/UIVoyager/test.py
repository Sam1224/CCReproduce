from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyTrajectoryDataset, collate_trajectories
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ToyTrajectoryDataset(num_samples=64)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_trajectories)

    model = OneModel().to(device)
    model.eval()

    acc_sum = 0.0
    n = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred = out.logits.argmax(dim=-1)
        mask = batch["actions"] != -100
        acc = (pred[mask] == batch["actions"][mask]).float().mean().item()
        acc_sum += acc
        n += 1

    print(f"Action accuracy (toy): {acc_sum / max(n, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
