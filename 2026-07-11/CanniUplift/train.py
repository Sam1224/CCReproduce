import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyMarketplaceUpliftDataset
from model import CanniUplift, canniuplift_loss


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = DataLoader(ToyMarketplaceUpliftDataset(length=args.samples), batch_size=args.batch_size, shuffle=True)
    model = CanniUplift().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["user"], batch["seller_features"], batch["treatment"], batch["coupon_value"])
            losses = canniuplift_loss(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            total_loss += losses["total"].item()
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(loader)):.4f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, output_dir / "canni_uplift_toy.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
