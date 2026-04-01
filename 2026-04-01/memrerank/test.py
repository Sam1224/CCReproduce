from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyMemRerankDataset
from model import MemRerankToyModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cpu")
    ds = ToyMemRerankDataset(n=8000, seed=2)
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = MemRerankToyModel().to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            pos, neg = model(
                history_item_ids=batch.history_item_ids.to(device),
                query_token_ids=batch.query_token_ids.to(device),
                pos_item_id=batch.pos_item_id.to(device),
                neg_item_id=batch.neg_item_id.to(device),
            )
            correct += (pos > neg).sum().item()
            total += pos.numel()

    print({"pairwise_acc": correct / max(1, total)})


if __name__ == "__main__":
    main()
