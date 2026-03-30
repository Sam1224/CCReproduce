from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import AuthorityRanker


def main() -> None:
    ckpt = torch.load("checkpoints/authoritybench.pt", map_location="cpu")
    model = AuthorityRanker()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    _, te = split(make_dataset(n=1200, seed=2), 0.0)

    correct = 0
    auth_sum = 0.0
    rel_sum = 0.0

    with torch.no_grad():
        for ex in te[:400]:
            scores = model(ex.feats.unsqueeze(0))[0]
            pred = int(torch.argmax(scores).item())
            gold = int(torch.argmax(ex.label).item())
            correct += 1 if pred == gold else 0
            rel_sum += float(ex.feats[pred, 0].item())
            auth_sum += float(ex.feats[pred, 1].item())

    n = 400
    print(f"top1_acc={correct/n:.3f} avg_rel={rel_sum/n:.3f} avg_auth={auth_sum/n:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
