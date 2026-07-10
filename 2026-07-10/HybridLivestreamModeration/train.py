import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ToyLivestreamModerationDataset
from model import HybridLivestreamModerationModel, hybrid_loss


def _move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = ToyLivestreamModerationDataset(length=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    reference_bank = dataset.build_reference_bank()
    reference_bank_tensors = {
        "frames": reference_bank.frames.to(device),
        "audio": reference_bank.audio.to(device),
        "text": reference_bank.text.to(device),
    }
    model = HybridLivestreamModerationModel(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(batch["frames"], batch["audio"], batch["text"], reference_bank_tensors)
            losses = hybrid_loss(outputs, batch, reference_bank.labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            running += losses["total"].item()
        mean_loss = running / max(1, len(loader))
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "reference_bank": {
                "frames": reference_bank.frames,
                "audio": reference_bank.audio,
                "text": reference_bank.text,
                "labels": reference_bank.labels,
            },
        },
        output_dir / "hybrid_livestream_moderation.pt",
    )
    print(f"saved checkpoint to {output_dir / 'hybrid_livestream_moderation.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
