from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import create_dataloaders
from model import GALA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_stage1", type=int, default=3)
    parser.add_argument("--epochs_stage2", type=int, default=2)
    parser.add_argument("--epochs_stage3", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="gala_toy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, triplet_loader, train_loader, test_loader = create_dataloaders(batch_size=args.batch_size)
    model = GALA(feature_dim=catalog.text_features.size(1), num_items=catalog.text_features.size(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    catalog_mm = 0.5 * (catalog.text_features + catalog.image_features)
    catalog_mm = catalog_mm.to(device)
    catalog_id = catalog.id_features.to(device)

    for epoch in range(args.epochs_stage1):
        model.train()
        running_loss = 0.0
        for batch in triplet_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model.stage1_triplet_loss(batch["query"], batch["positive"], batch["negative"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[stage1] epoch={epoch + 1} loss={running_loss / len(triplet_loader):.4f}")

    for epoch in range(args.epochs_stage2):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, metrics = model.stage2_alignment_loss(batch["history"], batch["query"], batch["target"], batch["reward"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += metrics["stage2_acc"]
        print(f"[stage2] epoch={epoch + 1} loss={running_loss / len(train_loader):.4f} acc={running_acc / len(train_loader):.4f}")

    for epoch in range(args.epochs_stage3):
        model.train()
        running_loss = 0.0
        running_recall = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, metrics = model.stage3_ranking_loss(
                batch["history"],
                batch["query"],
                batch["target"],
                catalog_mm,
                catalog_id,
                batch["reward"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_recall += metrics["recall@1"]
        print(f"[stage3] epoch={epoch + 1} loss={running_loss / len(train_loader):.4f} recall@1={running_recall / len(train_loader):.4f}")

    model.eval()
    metrics = {"recall@1": 0.0, "recall@5": 0.0}
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_metrics = model.evaluate(batch["history"], batch["query"], batch["target"], catalog_mm, catalog_id)
        for key, value in batch_metrics.items():
            metrics[key] += value
    metrics = {key: value / len(test_loader) for key, value in metrics.items()}
    print(f"[eval] {metrics}")

    checkpoint_path = Path(args.checkpoint)
    torch.save(
        {
            "model_state": model.state_dict(),
            "catalog_text": catalog.text_features,
            "catalog_image": catalog.image_features,
            "catalog_id": catalog.id_features,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
