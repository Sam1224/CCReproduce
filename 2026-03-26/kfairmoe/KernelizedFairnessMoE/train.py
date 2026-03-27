from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FairRecToyDataset, collate_fairrec
from model import OneModel, fairness_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FairRecToyDataset(num_samples=256, d_model=128, num_items=200, num_sensitive=2)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fairrec)

    model = OneModel(d_model=128, num_items=200, num_sensitive=2, num_experts=4, fairness_grl_scale=1.0).to(device)
    model.freeze_modules()
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(batch)
            loss, logs = fairness_loss(out, y=batch["y"], sensitive=batch["sensitive"], fairness_weight=1.0)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} total={loss.item():.4f} "
                    f"task={logs['task_loss']:.4f} leakage={logs['leakage_loss']:.4f}"
                )


if __name__ == "__main__":
    # Allow running as a script from anywhere.
    os.chdir(Path(__file__).resolve().parent)
    main()
