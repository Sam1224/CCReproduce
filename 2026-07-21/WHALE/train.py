import argparse
from pathlib import Path

import torch

from data import make_loaders
from model import WhaleToy, loss_fn


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)["logits"]
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == batch["label"]).sum().item()
            total += batch["label"].numel()
            probs.append(torch.sigmoid(logits).cpu())
            labels.append(batch["label"].cpu())
    return {"accuracy": correct / max(total, 1), "mean_score": torch.cat(probs).mean().item(), "positive_rate": torch.cat(labels).mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", type=str, default="checkpoints/whale.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.batch_size)
    model = WhaleToy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch={epoch} loss={total_loss/len(train_loader):.4f} metrics={evaluate(model, test_loader, device)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": evaluate(model, test_loader, device)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
