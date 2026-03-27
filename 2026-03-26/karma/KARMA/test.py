from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import KARMAToyDataset, collate_karma
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = KARMAToyDataset(num_samples=64, d_model=128, vocab_size=3000)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_karma)

    model = OneModel(d_model=128, vocab_size=3000).to(device)
    model.eval()

    correct = 0
    total = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        pred = (torch.sigmoid(out.action_logit) > 0.5).float()
        correct += int((pred == batch["action_label"]).sum().item())
        total += pred.numel()

    print(f"Action prediction accuracy (toy): {correct / max(total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
