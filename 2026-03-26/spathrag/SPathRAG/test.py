from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import KGQAToyDataset, collate_kgqa
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = KGQAToyDataset(num_samples=64, vocab_size=4096, num_paths=6)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_kgqa)

    model = OneModel(vocab_size=4096, d_model=256).to(device)
    model.eval()

    path_hits = 0
    total = 0

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred = (out.path_scores.argmax(dim=-1) == batch["path_labels"].argmax(dim=-1)).sum().item()
        path_hits += int(pred)
        total += batch["path_labels"].shape[0]

    print(f"Path selection accuracy@1 (toy): {path_hits / max(total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
