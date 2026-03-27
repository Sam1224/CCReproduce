from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyTrajectoryDataset, collate_trajectories
from model import OneModel, grsd_loss, rft_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ToyTrajectoryDataset(num_samples=256)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_trajectories)

    model = OneModel().to(device)
    optim = model.get_optim(lr=3e-4)

    # Stage 1: RFT
    model.train()
    for step, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss = rft_loss(out, actions=batch["actions"], mask=batch["mask"], traj_score=batch["traj_score"], keep_threshold=0.5)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 20 == 0:
            print(f"[RFT] step={step} loss={loss.item():.4f}")
        if step >= 60:
            break

    # Stage 2: GRSD
    for step, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss = grsd_loss(out, peer_logits=batch["peer_logits"], mask=batch["mask"], traj_score=batch["traj_score"], temperature=2.0)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 20 == 0:
            print(f"[GRSD] step={step} loss={loss.item():.4f}")
        if step >= 60:
            break


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
