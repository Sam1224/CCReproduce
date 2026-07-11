import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyMultimodalDocumentDataset
from model import CMDRChunkEncoder, cmdr_loss


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = DataLoader(ToyMultimodalDocumentDataset(length=args.samples), batch_size=args.batch_size, shuffle=True)
    model = CMDRChunkEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["query"], batch["page_text"], batch["page_images"])
            losses = cmdr_loss(outputs, batch["target"])
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            running += losses["total"].item()
        print(f"epoch={epoch + 1} loss={running / max(1, len(loader)):.4f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, output_dir / "cmdr_embed_toy.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
