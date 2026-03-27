from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MarchToyDataset, collate_march
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MarchToyDataset(num_samples=64)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_march)

    model = OneModel().to(device)
    model.eval()

    correct = 0
    total = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        pred = (torch.sigmoid(out.checker_logits) > 0.5).float()
        correct += int((pred == batch["claim_labels"]).sum().item())
        total += pred.numel()

    print(f"Claim supportedness accuracy (toy): {correct / max(total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
