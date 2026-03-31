from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ES3Builder, UniScaleDataset, collate_batch
from model import HHSFT
from train import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    builder = ES3Builder(seed=17)
    samples = builder.build_requests(n_requests=args.requests, candidates=args.candidates)
    ds = UniScaleDataset(samples)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)

    model = HHSFT().to(device)
    payload = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(payload["model"], strict=True)

    metrics = evaluate(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
