from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import MarchToyDataset, collate_march
from model import OneModel, march_losses


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MarchToyDataset(num_samples=256)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_march)

    model = OneModel().to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()

    # Phase 1: warm up checker + solver with supervised losses.
    for step, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss, logs = march_losses(out, response_tgt=batch["response_tgt"], claim_labels=batch["claim_labels"], reward_scale=0.0)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 20 == 0:
            print(f"[Warmup] step={step} total={loss.item():.4f} lm={logs['lm_loss']:.4f} chk={logs['checker_loss']:.4f}")
        if step >= 60:
            break

    # Phase 2: RL-style optimization (enable REINFORCE on proposer).
    for step, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss, logs = march_losses(out, response_tgt=batch["response_tgt"], claim_labels=batch["claim_labels"], reward_scale=1.0)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 20 == 0:
            print(
                f"[RL] step={step} total={loss.item():.4f} lm={logs['lm_loss']:.4f} "
                f"chk={logs['checker_loss']:.4f} rft={logs['reinforce']:.4f}"
            )
        if step >= 80:
            break


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
