import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ToyMarketplaceUpliftDataset
from model import CanniUplift


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CanniUplift().to(device)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    loader = DataLoader(ToyMarketplaceUpliftDataset(length=args.samples), batch_size=args.batch_size)
    model.eval()
    seller_mae = []
    platform_mae = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["user"], batch["seller_features"], batch["treatment"], batch["coupon_value"])
            seller_mae.append(F.l1_loss(outputs["seller_uplift"], batch["seller_uplift"]).item())
            platform_mae.append(F.l1_loss(outputs["platform_delta"], batch["platform_delta"]).item())
    print({"seller_uplift_mae": sum(seller_mae) / len(seller_mae), "platform_delta_mae": sum(platform_mae) / len(platform_mae)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/canni_uplift_toy.pt")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())
