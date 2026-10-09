from __future__ import annotations

import argparse
import json
import os

import torch

from data import build_prepared_data
from train import evaluate, make_dataset
from model import ModelConfig, OpticalRecModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.ckpt_dir, "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    prepared = build_prepared_data(seed=int(ckpt["seed"]))
    test_ds = make_dataset(prepared, "test", neg_seed=int(ckpt["seed"]) + 53)
    cfg = ModelConfig(**ckpt["cfg"])
    model = OpticalRecModel(cfg, mode=ckpt["mode"]).to(device)
    model.load_state_dict(ckpt["model"])

    item_bundle = {
        "cards": prepared.card_tensors.to(device),
        "images": prepared.image_tensors.to(device),
        "meta_tokens": prepared.meta_tokens.to(device),
        "meta_masks": prepared.meta_masks.to(device),
    }
    metrics = evaluate(model, test_ds, item_bundle, device=device, batch_size=args.batch_size)
    print(json.dumps({"ckpt": ckpt_path, **metrics}, indent=2))


if __name__ == "__main__":
    main()

