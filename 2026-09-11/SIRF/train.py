from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SirfToyDataset, collate_sirf
from model import OneModel, sirf_losses


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SirfToyDataset(num_samples=256)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_sirf)
    model = OneModel().to(device)
    optim = model.get_optim()

    model.train()
    for epoch in range(2):
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            cpt_weight = 1.0 if epoch == 0 else 0.35
            rationale_weight = 0.35 if epoch == 0 else 0.15
            loss, logs = sirf_losses(out, batch, cpt_weight=cpt_weight, rationale_weight=rationale_weight)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step} loss={logs['loss']:.4f} "
                    f"verdict={logs['verdict_loss']:.4f} cpt={logs['cpt_loss']:.4f} rationale={logs['rationale_loss']:.4f}"
                )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/sirf_toy.pt")
    print("saved checkpoints/sirf_toy.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
