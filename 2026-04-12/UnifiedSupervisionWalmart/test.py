from __future__ import annotations

import argparse

import numpy as np
import torch

from data import build_toy_dataset
from model import BiEncoderRetriever
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rel_eng.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")

    dataset = build_toy_dataset(seed=args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt["meta"]

    model = BiEncoderRetriever(vocab_size=meta["vocab_size"], dim=meta["dim"]).to(device)
    model.load_state_dict(ckpt["state"])

    m = evaluate(dataset=dataset, model=model, device=device)
    print(
        f"P@25={m['P@25']:.4f} NDCG@25={m['NDCG@25']:.4f} "
        f"AvgRel@25={m['AvgRel@25']:.4f} AvgEng@25={m['AvgEng@25']:.4f}"
    )


if __name__ == "__main__":
    main()
