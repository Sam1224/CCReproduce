from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ProductFamilyToyDataset, collate_ptc
from model import OneModel, ptc_losses


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProductFamilyToyDataset(num_samples=384)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_ptc)
    model = OneModel().to(device)
    optim = model.get_optim()

    model.train()
    for epoch in range(3):
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            loss, logs = ptc_losses(out, batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if step % 15 == 0:
                print(f"epoch={epoch} step={step} loss={logs['loss']:.4f} ce={logs['ce']:.4f} kl={logs['teacher_kl']:.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/ptc_toy.pt")
    print("saved checkpoints/ptc_toy.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
