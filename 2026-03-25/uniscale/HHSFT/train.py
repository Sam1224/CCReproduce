from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyRankingDataset, collate_ranking
from model import OneModel, ranking_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ToyRankingDataset(num_samples=512)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_ranking)

    model = OneModel().to(device)
    optim = model.get_optim(lr=3e-4)

    model.train()
    for epoch in range(1):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, logs = ranking_loss(out, label=batch["label"])

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                pair_acc = float((out.pos_score > out.neg_score).float().mean().detach())
                print(
                    f"epoch={epoch} step={step} total={loss.item():.4f} "
                    f"point={logs['point_loss']:.4f} pair={logs['pair_loss']:.4f} pair_acc={pair_acc:.3f}"
                )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
