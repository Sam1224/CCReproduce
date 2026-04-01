from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyModerationDataset
from model import XuanwuToyMMClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cpu")

    ds = ToyModerationDataset(n=2000, seed=2)
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = XuanwuToyMMClassifier().to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(image_feat=batch.image_feat.to(device), token_ids=batch.token_ids.to(device))
            pred = logits.argmax(dim=-1)
            correct += (pred.cpu() == batch.label).sum().item()
            total += batch.label.numel()

    print({"acc": correct / max(1, total)})


if __name__ == "__main__":
    main()
