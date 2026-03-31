from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import BLMGuardDataset, collate
from model import BLMGuardModel
from train import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ds = BLMGuardDataset(n=1024, seed=17)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.ckpt, map_location=device)
    vocab = payload.get("vocab")
    vocab_size = len(vocab) if isinstance(vocab, dict) else len(ds.vocab.stoi)

    model = BLMGuardModel(vocab=vocab_size).to(device)
    model.load_state_dict(payload["model"], strict=True)

    metrics = evaluate(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
