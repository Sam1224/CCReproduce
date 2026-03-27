from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FairRecToyDataset, collate_fairrec
from model import OneModel


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FairRecToyDataset(num_samples=64, d_model=128, num_items=200, num_sensitive=2)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fairrec)

    model = OneModel(d_model=128, num_items=200, num_sensitive=2).to(device)
    model.freeze_modules()
    model.eval()

    task_correct = 0
    task_total = 0
    sens_correct = 0
    sens_total = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)

        y = batch["y"]
        sensitive = batch["sensitive"]

        y_pred = out.task_logits.argmax(dim=-1)
        s_pred = out.sensitive_logits.argmax(dim=-1)

        mask = y != -100
        task_correct += (y_pred[mask] == y[mask]).sum().item()
        task_total += mask.sum().item()

        mask_s = sensitive != -100
        sens_correct += (s_pred[mask_s] == sensitive[mask_s]).sum().item()
        sens_total += mask_s.sum().item()

    print(f"Task accuracy (toy): {task_correct / max(task_total, 1):.3f}")
    print(f"Sensitive accuracy / leakage (toy): {sens_correct / max(sens_total, 1):.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
