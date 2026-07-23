from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import create_dataloaders
from model import GenerativeSlateModel, training_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def evaluate(model: GenerativeSlateModel, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            user_state = batch["user_state"].to(device)
            constraints = batch["constraints"].to(device)
            targets = batch["targets"].to(device)
            logits = model(user_state, constraints, targets)
            loss, _ = training_loss(logits, targets)
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(1, total_batches)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog, train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        slate_size=5,
        seed=args.seed,
    )
    model = GenerativeSlateModel(
        feature_dim=catalog.item_features.size(1),
        hidden_dim=args.hidden_dim,
        num_items=catalog.item_features.size(0),
        slate_size=5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "spd_decoding.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_batches = 0

        for batch in train_loader:
            user_state = batch["user_state"].to(device)
            constraints = batch["constraints"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits = model(user_state, constraints, targets)
            loss, stats = training_loss(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_acc += stats["token_accuracy"]
            total_batches += 1

        val_loss = evaluate(model, test_loader, device)
        train_loss = running_loss / max(1, total_batches)
        train_acc = running_acc / max(1, total_batches)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_token_acc={train_acc:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "catalog": {
                        "item_features": catalog.item_features,
                        "item_categories": catalog.item_categories,
                        "creator_groups": catalog.creator_groups,
                        "business_scores": catalog.business_scores,
                    },
                    "config": {
                        "feature_dim": catalog.item_features.size(1),
                        "num_items": catalog.item_features.size(0),
                        "hidden_dim": args.hidden_dim,
                        "slate_size": 5,
                        "seed": args.seed,
                    },
                },
                checkpoint_path,
            )

    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
