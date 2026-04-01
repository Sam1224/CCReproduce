from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyMMSRDPODataset
from model import RDPO_MoE_ToyModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cpu")

    ds = ToyMMSRDPODataset(n=10000, seed=2)
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = RDPO_MoE_ToyModel().to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            pos, neg = model(
                seq_item_ids=batch.seq_item_ids.to(device),
                seq_modality_ids=batch.seq_modality_ids.to(device),
                pos_next=batch.pos_next.to(device),
                neg_next=batch.neg_next.to(device),
            )
            correct += (pos > neg).sum().item()
            total += pos.numel()

    print({"pairwise_acc": correct / max(1, total)})


if __name__ == "__main__":
    main()
