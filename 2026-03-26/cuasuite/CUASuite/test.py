from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ScreenshotCuaToyDataset, collate_cua
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ScreenshotCuaToyDataset(num_samples=64)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_cua)

    model = OneModel(d_model=128, num_actions=12).to(device)
    model.eval()

    correct = 0
    total = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred = out.action_logits.argmax(dim=-1)
        correct += int((pred == batch["action_id"]).sum().item())
        total += pred.numel()

    print(f"Action accuracy (toy): {correct / max(total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
