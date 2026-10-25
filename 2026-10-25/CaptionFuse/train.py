from __future__ import annotations

import argparse
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset import build_catalog, iter_titles
from model import ProductEncoder, Vocabulary, pad_text_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output", type=str, default="checkpoints/toy_encoder.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = build_catalog()
    titles = list(iter_titles(catalog))
    vocab = Vocabulary(titles)
    encoded = [vocab.encode(title) for title in titles]

    model = ProductEncoder(len(vocab), len(catalog), dim=args.dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dataset = TensorDataset(torch.arange(len(catalog), dtype=torch.long))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        losses = []
        for (product_ids,) in loader:
            product_ids = product_ids.to(args.device)
            batch_text = [encoded[i] for i in product_ids.cpu().tolist()]
            padded, mask = pad_text_batch(batch_text)
            padded = padded.to(args.device)
            mask = mask.to(args.device)
            logits = model(product_ids, padded, mask)
            target = torch.arange(logits.shape[0], device=args.device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1:03d} loss={sum(losses) / len(losses):.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "dim": args.dim,
            "num_products": len(catalog),
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
