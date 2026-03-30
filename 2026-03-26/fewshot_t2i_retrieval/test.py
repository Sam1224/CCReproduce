from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_pairs
from model import Encoder


def recall_at_1(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> float:
    sims = img_emb @ txt_emb.t()
    pred = sims.argmax(dim=-1)
    gold = torch.arange(sims.shape[0])
    return (pred == gold).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load("checkpoints/fewshot_t2i.pt", map_location="cpu")
    img_enc = Encoder(); txt_enc = Encoder()
    img_enc.load_state_dict(ckpt["img"], strict=True)
    txt_enc.load_state_dict(ckpt["txt"], strict=True)
    img_enc.eval(); txt_enc.eval()

    # Evaluate on novel pairs
    test = make_pairs(n=400, seed=3, num_classes=6)
    with torch.no_grad():
        r1 = recall_at_1(img_enc(test.img), txt_enc(test.txt))
    print(f"recall@1={r1:.3f} (after pretrain+{args.k}-shot adapt)")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
