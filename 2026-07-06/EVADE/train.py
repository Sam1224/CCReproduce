import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import EvadeDataset, collate_batch
from model import EvadeModerator


def train(args):
    dataset = EvadeDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    device = torch.device(args.device)
    model = EvadeModerator(vocab_size=len(dataset.vocab.token_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / max(len(loader), 1):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": dataset.vocab.token_to_id}, output_dir / "evade.pt")
    print(f"saved checkpoint to {output_dir / 'evade.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    train(parser.parse_args())
