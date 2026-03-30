from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import load_dataset
from model import TextEncoder, contrastive_loss, make_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--in-dim", type=int, default=4096)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="checkpoints/retriever.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset(args.data)

    chunk_map = {c.chunk_id: c.text for c in ds.chunks}
    pairs = [(e.question, chunk_map.get(e.support_chunk_id, "")) for e in ds.examples if e.support_chunk_id in chunk_map]

    model = TextEncoder(in_dim=args.in_dim, out_dim=args.out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for i in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[i : i + args.batch_size]
            if not batch_pairs:
                continue
            questions = [p[0] for p in batch_pairs]
            docs = [p[1] for p in batch_pairs]

            b = make_batch(questions, docs, in_dim=args.in_dim)
            q = b.q.to(device)
            d = b.d_pos.to(device)

            q_emb = model(q)
            d_emb = model(d)
            loss = contrastive_loss(q_emb, d_emb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.detach())
            n += 1

        avg = total / max(1, n)
        print(f"epoch={epoch} loss={avg:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": args.in_dim,
            "out_dim": args.out_dim,
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
