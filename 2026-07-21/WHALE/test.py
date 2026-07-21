import argparse

import torch

from data import make_loaders
from model import WhaleToy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/whale.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = make_loaders(batch_size=64)
    model = WhaleToy().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    correct = 0
    total = 0
    score_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)["logits"]
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            correct += (pred == batch["label"]).sum().item()
            total += batch["label"].numel()
            score_sum += probs.mean().item()

    print({
        "accuracy": correct / max(total, 1),
        "mean_prediction_score": score_sum / max(len(loader), 1),
        "saved_metrics": ckpt.get("metrics", {}),
    })


if __name__ == "__main__":
    main()
