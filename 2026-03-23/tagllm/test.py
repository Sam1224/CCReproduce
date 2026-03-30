from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import TAGS, build_dataset, split
from model import Tagger


def micro_f1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    pred = pred.bool()
    gold = gold.bool()
    tp = (pred & gold).sum().item()
    fp = (pred & (~gold)).sum().item()
    fn = ((~pred) & gold).sum().item()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    ds = build_dataset(n=1500, seed=2)
    _, (xte, yte) = split(ds, 0.0)

    ckpt = torch.load("checkpoints/tagllm.pt", map_location="cpu")
    model = Tagger(vocab_size=len(ckpt["vocab"]), num_tags=len(TAGS))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        out = torch.sigmoid(model(xte))
    f1 = micro_f1(out > 0.5, yte)
    print(f"microF1={f1:.3f} tags={TAGS}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
