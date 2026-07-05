import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ToyFashionDetailDataset
from model import DetailAnywhere, ViewBridgingTeacher, detailanywhere_loss


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = ToyFashionDetailDataset(length=args.samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DetailAnywhere().to(device)
    teacher = ViewBridgingTeacher().to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            reference = batch["reference"].to(device)
            focus_mask = batch["focus_mask"].to(device)
            target_detail = batch["target_detail"].to(device)
            category_id = batch["category_id"].to(device)
            with torch.no_grad():
                teacher_feature = teacher(reference, focus_mask)
            outputs = model(reference, focus_mask, category_id)
            losses = detailanywhere_loss(outputs, target_detail, teacher_feature)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            running_loss += losses["total"].item()
        mean_loss = running_loss / max(1, len(loader))
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")

    torch.save({"model": model.state_dict()}, output_dir / "detailanywhere_toy.pt")
    print(f"saved checkpoint to {output_dir / 'detailanywhere_toy.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
