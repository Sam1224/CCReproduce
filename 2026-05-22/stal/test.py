import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyAIGenDataset
from model import SpatialDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SpatialDetector(dim=128)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = ToyAIGenDataset(n=400, image_size=64, seed=123)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            _, logit = model(x)
            pred = (torch.sigmoid(logit) > 0.5).long()
            correct += int((pred == y).sum().item())
            total += y.numel()

    print(f"spatial-only accuracy: {correct/total:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
