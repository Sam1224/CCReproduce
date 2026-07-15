import argparse

import torch

from data import make_loaders
from model import TwoTower, TwoTowerConfig, hit_rate_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/goobs.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, num_users, num_items = make_loaders(batch_size=128)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = TwoTower(TwoTowerConfig(ckpt.get("num_users", num_users), ckpt.get("num_items", num_items))).to(device)
    model.load_state_dict(ckpt["model"])
    print({"hr@50": hit_rate_at_k(model, test_loader, num_items, 50, device), "hr@100": hit_rate_at_k(model, test_loader, num_items, 100, device)})


if __name__ == "__main__":
    main()
