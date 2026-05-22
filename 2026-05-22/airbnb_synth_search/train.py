import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TriplesDataset, Vocab, collate_batch
from model import DualEncoder, pairwise_ranking_loss


def build_vocab(train_path: str) -> Vocab:
    texts = []
    with Path(train_path).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.extend([r["query"], r["pos_text"], r["neg_text"]])
    return Vocab.build(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="checkpoints/model.pt")
    args = parser.parse_args()

    vocab = build_vocab(args.train)
    ds = TriplesDataset(args.train, vocab)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_batch)

    model = DualEncoder(vocab_size=len(vocab.token_to_id), dim=args.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for (q_flat, q_off), (p_flat, p_off), (n_flat, n_off) in dl:
            q = model.encode(q_flat, q_off)
            p = model.encode(p_flat, p_off)
            n = model.encode(n_flat, n_off)
            pos = model.score(q, p)
            neg = model.score(q, n)
            loss = pairwise_ranking_loss(pos, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch}: loss={total/len(dl):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.token_to_id, "dim": args.dim}, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
