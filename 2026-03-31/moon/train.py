from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_pairs, make_synthetic_catalog, split
from model import MoonEncoder, info_nce


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    products, meta = make_synthetic_catalog(n_products=4500, seed=0)
    pairs = make_pairs(products, n_pairs=7000, seed=1)
    tr, _ = split(pairs, 0.9)

    model = MoonEncoder(vocab_size=meta["vocab_size"], d_model=128, d_img=meta["d_img"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0

        for i in range(0, len(tr), args.batch):
            batch = tr[i : i + args.batch]
            if len(batch) < 2:
                continue

            q = []
            k = []
            for ex in batch:
                q.append(model.encode_text(ex.query_token_ids, ex.query_token_types))
                p = ex.product
                k.append(model.encode_product(p.token_ids, p.token_types, p.images_full, p.images_core))

            q_t = torch.stack(q, dim=0)
            k_t = torch.stack(k, dim=0)

            loss = info_nce(q_t, k_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach())
            steps += 1

        print(f"epoch={ep} loss={total/max(1,steps):.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt / "moon.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
