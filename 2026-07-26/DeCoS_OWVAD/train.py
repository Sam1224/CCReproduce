import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from dataset import DeCoSToyDataset, build_definition_bank
from model import DeCoSScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy DeCoS scorer.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="decos_toy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    definition_bank = build_definition_bank(args.num_classes, args.feature_dim, seed=args.seed)
    train_dataset = DeCoSToyDataset(
        num_samples=args.train_samples,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        definition_bank=definition_bank,
        seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = DeCoSScorer(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        shared_direction=definition_bank.shared_direction,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    anomaly_embeddings = definition_bank.anomaly_embeddings.float()
    normal_embedding = definition_bank.normal_embedding.float()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        counted_batches = 0

        for batch in train_loader:
            visual_features = batch["visual_features"].float()
            frame_labels = batch["frame_labels"].long()
            result = model(visual_features, anomaly_embeddings, normal_embedding)
            scores = result["scores"]

            anomaly_mask = frame_labels > 0
            if anomaly_mask.sum() == 0:
                continue

            logits = scores[anomaly_mask]
            targets = frame_labels[anomaly_mask] - 1
            loss = functional.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            counted_batches += 1

        mean_loss = running_loss / max(counted_batches, 1)
        print(f"epoch={epoch + 1:02d} loss={mean_loss:.4f}")

    output_path = Path(args.output)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "feature_dim": args.feature_dim,
                "hidden_dim": args.hidden_dim,
                "num_classes": args.num_classes,
                "seq_len": args.seq_len,
                "seed": args.seed,
            },
            "definition_bank": {
                "normal_embedding": definition_bank.normal_embedding,
                "anomaly_embeddings": definition_bank.anomaly_embeddings,
                "shared_direction": definition_bank.shared_direction,
                "class_directions": definition_bank.class_directions,
            },
        },
        output_path,
    )
    print(f"saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
