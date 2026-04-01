from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TAXONOMY, SyntheticUser, build_synthetic_users
from model import InterestExtractor


def build_leaf_vocab() -> list[str]:
    leafs: list[str] = []
    for parent, children in TAXONOMY["root"].items():
        for leaf in children:
            leafs.append(f"{parent}/{leaf}")
    return sorted(leafs)


def build_interest_vocab() -> list[str]:
    return sorted(TAXONOMY["root"].keys())


def collate(batch: list[SyntheticUser], *, leaf_index: dict[str, int], interest_index: dict[str, int], max_len: int):
    leaf_ids_list = []
    weight_list = []
    mask_list = []
    label_list = []

    for user in batch:
        leaf_ids = []
        weights = []
        for interaction in user.history[:max_len]:
            leaf_ids.append(leaf_index[interaction.category])
            weights.append(float(interaction.weight))

        mask = [1.0] * len(leaf_ids)
        while len(leaf_ids) < max_len:
            leaf_ids.append(0)
            weights.append(0.0)
            mask.append(0.0)

        labels = [0.0] * len(interest_index)
        for interest in user.gt_interests:
            if interest in interest_index:
                labels[interest_index[interest]] = 1.0

        leaf_ids_list.append(torch.tensor(leaf_ids, dtype=torch.long))
        weight_list.append(torch.tensor(weights, dtype=torch.float32))
        mask_list.append(torch.tensor(mask, dtype=torch.float32))
        label_list.append(torch.tensor(labels, dtype=torch.float32))

    return (
        torch.stack(leaf_ids_list, dim=0),
        torch.stack(weight_list, dim=0),
        torch.stack(mask_list, dim=0),
        torch.stack(label_list, dim=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--out", default="ckpt.pt")
    args = ap.parse_args()

    users = build_synthetic_users(num_users=160, seed=0)
    leaf_vocab = build_leaf_vocab()
    interest_vocab = build_interest_vocab()

    leaf_index = {leaf: idx for idx, leaf in enumerate(leaf_vocab)}
    interest_index = {interest: idx for idx, interest in enumerate(interest_vocab)}

    model = InterestExtractor(num_leaf_categories=len(leaf_vocab), num_interests=len(interest_vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        users,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate(b, leaf_index=leaf_index, interest_index=interest_index, max_len=args.max_len),
    )

    model.train()
    for _ in range(args.epochs):
        for leaf_ids, weights, mask, labels in loader:
            out = model(leaf_ids, weights, mask)
            loss = loss_fn(out.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    out_path = Path(args.out)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "leaf_vocab": leaf_vocab,
            "interest_vocab": interest_vocab,
            "max_len": args.max_len,
        },
        out_path,
    )
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
