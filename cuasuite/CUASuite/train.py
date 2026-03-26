from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VideoCuaToyDataset, collate_cua
from model import OneModel, imitation_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = VideoCuaToyDataset(num_samples=256)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_cua)

    model = OneModel(d_model=128, num_actions=12).to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, logs = imitation_loss(out, action_id=batch["action_id"])

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
