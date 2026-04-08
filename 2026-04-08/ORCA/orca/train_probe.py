from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticDataConfig, make_splits
from .models import ConfidenceProbe
from .utils import default_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_instances", type=int, default=8000)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = default_device()

    cfg = SyntheticDataConfig(
        num_instances=args.num_instances,
        max_steps=args.max_steps,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )
    train_ds, _, _ = make_splits(cfg)

    model = ConfidenceProbe(feature_dim=args.feature_dim, hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            logits = model(x)
            loss = (bce(logits, y) * mask).sum() / mask.sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

    ckpt_path = os.path.join(args.out_dir, "probe.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
        },
        ckpt_path,
    )
    print(f"[saved] {ckpt_path}")


if __name__ == "__main__":
    main()
