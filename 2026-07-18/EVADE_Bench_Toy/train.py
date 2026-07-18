import argparse
from pathlib import Path

import torch

from data import make_loaders
from model import EvadeBenchToyModel, binary_accuracy, loss_fn, mask_iou


def evaluate(model, loader, device):
    model.eval()
    category_correct = 0
    total_examples = 0
    violation_acc = 0.0
    consistency_acc = 0.0
    evasion_iou = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch)
            category_correct += (outputs["category_logits"].argmax(dim=-1) == batch["label"]).sum().item()
            total_examples += batch["label"].numel()
            violation_acc += binary_accuracy(outputs["violation_logits"], batch["violation"])
            consistency_acc += binary_accuracy(outputs["consistency_logits"], batch["consistency"])
            evasion_iou += mask_iou(outputs["evasion_logits"], batch["evasion_mask"])
    batches = max(len(loader), 1)
    return {
        "category_accuracy": category_correct / max(total_examples, 1),
        "violation_accuracy": violation_acc / batches,
        "cross_modal_consistency_accuracy": consistency_acc / batches,
        "evasion_mask_iou": evasion_iou / batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", type=str, default="checkpoints/evade_bench_toy.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.batch_size)
    model = EvadeBenchToyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics = evaluate(model, test_loader, device)
        print(f"epoch={epoch} loss={total_loss / len(train_loader):.4f} metrics={metrics}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": evaluate(model, test_loader, device)}, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
