import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyMultimodalDocumentDataset
from model import CMDRChunkEncoder


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CMDRChunkEncoder().to(device)
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    loader = DataLoader(ToyMultimodalDocumentDataset(length=args.samples), batch_size=args.batch_size)
    hits = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = model(batch["query"], batch["page_text"], batch["page_images"])["scores"]
            hits += (scores.argmax(dim=1) == batch["target"]).sum().item()
            total += batch["target"].numel()
    print({"toy_top1_accuracy": hits / max(1, total)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/cmdr_embed_toy.pt")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())
