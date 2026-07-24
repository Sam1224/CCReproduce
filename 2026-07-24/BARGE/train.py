from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import create_dataloaders
from model import BARGEModel, training_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: BARGEModel, catalog, loader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    plain_hits = 0
    barge_hits = 0
    total = 0
    for batch in loader:
        history = batch["history"].to(device)
        main_codes = batch["main_codes"].to(device)
        aux_codes = batch["aux_codes"].to(device)
        target_item = batch["target_item"].to(device)
        outputs = model(history, main_codes, aux_codes)
        loss, _ = training_loss(outputs, main_codes, aux_codes)
        total_loss += float(loss.item())
        total_batches += 1

        plain = model.decode_plain(history, catalog)
        barge = model.decode_barge(history, catalog)
        plain_hits += int((plain.items == target_item).sum().item())
        barge_hits += int((barge.items == target_item).sum().item())
        total += int(target_item.numel())
    return {
        "val_loss": total_loss / max(1, total_batches),
        "plain_recall@1": plain_hits / max(1, total),
        "barge_recall@1": barge_hits / max(1, total),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog, train_loader, val_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = BARGEModel(
        num_items=catalog.num_items,
        codebook_size=catalog.codebook_size,
        hidden_dim=args.hidden_dim,
        history_len=catalog.history_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "barge_toy.pt"
    best_gap = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_main = 0.0
        running_aux = 0.0
        steps = 0
        for batch in train_loader:
            history = batch["history"].to(device)
            main_codes = batch["main_codes"].to(device)
            aux_codes = batch["aux_codes"].to(device)

            optimizer.zero_grad()
            outputs = model(history, main_codes, aux_codes)
            loss, stats = training_loss(model, outputs, main_codes, aux_codes)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_main += stats["main_token_acc"]
            running_aux += stats["aux_token_acc"]
            steps += 1

        metrics = evaluate(model, catalog, val_loader, device)
        gain = metrics["barge_recall@1"] - metrics["plain_recall@1"]
        print(
            f"epoch={epoch} train_loss={running_loss/max(1,steps):.4f} main_acc={running_main/max(1,steps):.4f} "
            f"aux_acc={running_aux/max(1,steps):.4f} val_loss={metrics['val_loss']:.4f} "
            f"plain_r1={metrics['plain_recall@1']:.4f} barge_r1={metrics['barge_recall@1']:.4f} gain={gain:.4f}"
        )
        if gain > best_gap:
            best_gap = gain
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "catalog": {
                        "item_features": catalog.item_features,
                        "item_categories": catalog.item_categories,
                        "popularity": catalog.popularity,
                        "main_codes": catalog.main_codes,
                        "aux_codes": catalog.aux_codes,
                        "codebook_size": catalog.codebook_size,
                        "history_len": catalog.history_len,
                    },
                    "config": {
                        "num_items": catalog.num_items,
                        "hidden_dim": args.hidden_dim,
                        "codebook_size": catalog.codebook_size,
                        "history_len": catalog.history_len,
                        "seed": args.seed,
                    },
                },
                checkpoint_path,
            )

    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
