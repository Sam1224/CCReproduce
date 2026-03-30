from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import issue_to_onehot, load_examples
from model import Ranker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    examples = load_examples(args.train)
    model = Ranker()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for ex in examples:
            mem = torch.tensor(ex.memory.to_vec(), dtype=torch.float32)
            iss = torch.tensor(issue_to_onehot(ex.issue_type), dtype=torch.float32)

            feats = []
            for c in ex.candidates:
                pr = torch.tensor(c.to_vec(), dtype=torch.float32)
                feats.append(torch.cat([mem, iss, pr], dim=0))
            X = torch.stack(feats, dim=0)  # (K, D)

            scores = model(X)  # (K,)
            y = torch.tensor([ex.label], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach())
            n += 1

        print(f"epoch={ep} loss={total/max(1,n):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
