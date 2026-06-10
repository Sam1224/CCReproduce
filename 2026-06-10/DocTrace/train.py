import argparse
import json
import os

import numpy as np
import torch

from agents import FilterModel, hash_vector


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def make_training_examples(records, dim: int, max_paras_per_doc: int | None):
    xs = []
    ys = []
    for r in records:
        query = r["query"]
        gold_pid = int(r["gold_pid"])
        paragraphs = r["paragraphs"]
        if max_paras_per_doc is not None:
            paragraphs = paragraphs[:max_paras_per_doc]
        for pid, p in enumerate(paragraphs):
            x = hash_vector(query + " [SEP] " + p, dim=dim)
            y = 1.0 if pid == gold_pid else 0.0
            xs.append(x)
            ys.append(y)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_paras_per_doc", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    records = list(load_jsonl(train_path))
    x, y = make_training_examples(records, dim=args.dim, max_paras_per_doc=args.max_paras_per_doc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FilterModel(dim=args.dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    idx = torch.arange(x.size(0))

    for epoch in range(1, args.epochs + 1):
        perm = idx[torch.randperm(idx.size(0))]
        total = 0.0
        model.train()

        for start in range(0, perm.size(0), args.batch_size):
            b = perm[start : start + args.batch_size]
            xb = x[b].to(device)
            yb = y[b].to(device)

            logit = model(xb)
            loss = loss_fn(logit, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * xb.size(0)

        avg = total / perm.size(0)
        print(f"epoch={epoch:02d} loss={avg:.4f}")

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save({"model": model.state_dict(), "dim": args.dim}, ckpt_path)
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
