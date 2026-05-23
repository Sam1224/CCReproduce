from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyEmbeddingDataset
from model import CEDAR, reconstruction_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ds = ToyEmbeddingDataset(num_samples=8192, embedding_dim=args.dim, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = CEDAR(dim=args.dim, k=args.k).to(args.device)

    # Initialize mean with one batch
    first = next(iter(dl))["z"].to(args.device)
    model.init_mean(first)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for batch in dl:
            z = batch["z"].to(args.device)
            out = model(z)
            loss = reconstruction_loss(out.z_hat, z, kind="l1")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu()) * z.shape[0]

        avg = running / len(ds)
        if epoch == 1 or epoch % 20 == 0:
            active = float(model(z).active_mask.float().mean().cpu())
            print(f"epoch={epoch:04d} loss={avg:.6f} active_ratio={active:.4f}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "cedar.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dim": args.dim,
            "k": args.k,
            "mean": model.mean.detach().cpu(),
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
