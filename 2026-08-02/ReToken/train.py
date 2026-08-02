import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticReTokenConfig, SyntheticVisualHaystackDataset
from model import ReTokenRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight ReToken reproduction on a toy visual-haystack task.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="checkpoints/retoken.pt")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticVisualHaystackDataset(SyntheticReTokenConfig())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ReTokenRetriever().to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            frame_features = batch["frame_features"].to(device)
            question_features = batch["question_features"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(frame_features, question_features, labels)
            outputs["loss"].backward()
            optimizer.step()
            total_loss += outputs["loss"].item() * frame_features.size(0)
        print(f"epoch={epoch + 1} loss={total_loss / len(dataset):.4f}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args)}, output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    train()
