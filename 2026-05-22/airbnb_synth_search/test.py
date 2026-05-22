import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TriplesDataset, Vocab, collate_batch, kl_divergence, length_histogram
from model import DualEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = Vocab(token_to_id=ckpt["vocab"])
    model = DualEncoder(vocab_size=len(vocab.token_to_id), dim=int(ckpt["dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = TriplesDataset(args.data, vocab)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)

    correct = 0
    total = 0
    with torch.no_grad():
        for (q_flat, q_off), (p_flat, p_off), (n_flat, n_off) in dl:
            q = model.encode(q_flat, q_off)
            p = model.encode(p_flat, p_off)
            n = model.encode(n_flat, n_off)
            pos = model.score(q, p)
            neg = model.score(q, n)
            correct += int((pos > neg).sum().item())
            total += pos.shape[0]

    acc = correct / max(1, total)
    print(f"pairwise accuracy: {acc:.3f} ({correct}/{total})")

    # A tiny 'real user' proxy: the seed queries themselves.
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    synthetic_queries = [r["query"] for r in rows]
    seed_queries = [
        line.strip()
        for line in Path("toy_data/seed_queries.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    p = length_histogram(synthetic_queries)
    q = length_histogram(seed_queries)
    print(f"KL(length || seed) = {kl_divergence(p, q):.3f}")


if __name__ == "__main__":
    main()
