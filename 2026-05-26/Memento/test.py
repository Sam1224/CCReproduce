from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from data import ToyAdsConfig, ToyAdsDataset
from model import CTRModel, aggregate_history_lastn, aggregate_history_mmr
from train import eval_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]

    cfg = ToyAdsConfig(dim=int(train_args["dim"]), history_len=int(train_args["history_len"]), seed=int(train_args["seed"]))
    test_ds = ToyAdsDataset(cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=int(train_args["batch_size"]))

    model = CTRModel(dim=int(train_args["dim"]))
    model.load_state_dict(ckpt["model"])
    model.to(args.device)

    metrics = eval_split(
        model,
        test_loader,
        mode=str(train_args["mode"]),
        k=int(train_args["k"]),
        lambda_relevance=float(train_args["lambda_relevance"]),
        device=args.device,
    )

    print(metrics)


if __name__ == "__main__":
    main()
