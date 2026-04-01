from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import SimulaToyDataset
from model import SimulaToyModel


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cpu")

    test_easy = SimulaToyDataset(n=6000, seed=2, hard=False)
    test_hard = SimulaToyDataset(n=6000, seed=3, hard=True)

    model = SimulaToyModel().to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    for name, ds in [("easy", test_easy), ("hard", test_hard)]:
        loader = DataLoader(ds, batch_size=args.batch_size)
        preds = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                preds.append(model(batch.feats.to(device)).cpu())
                targets.append(batch.target)
        print({name: {"mse": mse(torch.cat(preds, 0), torch.cat(targets, 0))}})


if __name__ == "__main__":
    main()
