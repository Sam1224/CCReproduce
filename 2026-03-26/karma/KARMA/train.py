from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import KARMAToyDataset, collate_karma
from model import OneModel, karma_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = KARMAToyDataset(num_samples=256, d_model=128, vocab_size=3000)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_karma)

    model = OneModel(d_model=128, vocab_size=3000).to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, logs = karma_loss(
                out,
                action_label=batch["action_label"],
                sem_ids=batch["sem_ids"],
                sem_vec=batch["sem_vec"],
                alpha=1.0,
                beta=1.0,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} total={loss.item():.4f} "
                    f"act={logs['action_loss']:.4f} gen={logs['gen_loss']:.4f} rec={logs['recon_loss']:.4f}"
                )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
