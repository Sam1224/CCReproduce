from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    ds = build_dataset(n=3000, seed=1)
    (xtr, ytr), (xte, yte) = split(ds, 0.85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tagger(vocab_size=len(ds.vocab), num_tags=len(TAGS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        logits = model(xtr.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = torch.sigmoid(model(xte.to(device))).cpu()
            f1 = micro_f1(out > args.threshold, yte)
        print(f"epoch={ep} loss={loss.item():.4f} microF1={f1:.3f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": ds.vocab}, ckpt / "tagllm.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
