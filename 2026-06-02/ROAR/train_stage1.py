from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from data import build_toy_ecommerce_dataset
from losses import contrastive_loss_with_false_negative_mask
from model import DualEncoder, SimpleTokenizer


def _collate(tokenizer: SimpleTokenizer, batch) -> tuple:
    queries = [ex.query for ex in batch]
    products = [ex.product for ex in batch]
    q_batch = tokenizer.encode_batch(queries)
    d_batch = tokenizer.encode_batch(products)
    return q_batch, d_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_toy_ecommerce_dataset()

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([ex.query for ex in bundle.stage1_train] + [ex.product for ex in bundle.stage1_train])

    model = DualEncoder(vocab_size=len(tokenizer.vocab), hidden_size=256).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dl = DataLoader(
        bundle.stage1_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(tokenizer, b),
        drop_last=True,
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for q_batch, d_batch in dl:
            q_batch.token_ids = q_batch.token_ids.to(device)
            q_batch.attn_mask = q_batch.attn_mask.to(device)
            d_batch.token_ids = d_batch.token_ids.to(device)
            d_batch.attn_mask = d_batch.attn_mask.to(device)

            q, d = model(q_batch, d_batch)
            loss = contrastive_loss_with_false_negative_mask(q, d, margin_delta=args.delta)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.detach().cpu())

        print(f"epoch={epoch} loss={total/len(dl):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "tokenizer_vocab": tokenizer.vocab,
        "model_state": model.state_dict(),
        "config": {"hidden_size": 256},
    }
    out_path = out_dir / "stage1.pt"
    torch.save(ckpt, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
