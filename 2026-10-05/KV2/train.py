import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NeedleDataset
from model import TinyKVModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", default="runs/demo")
    parser.add_argument("--seq-len", type=int, default=96)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NeedleDataset(size=2048, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = TinyKVModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(loader)):.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "seq_len": args.seq_len}, os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    main()
