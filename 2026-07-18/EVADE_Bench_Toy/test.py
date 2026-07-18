import argparse

import torch

from data import make_loaders
from model import EvadeBenchToyModel, binary_accuracy, mask_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/evade_bench_toy.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = make_loaders(batch_size=32)
    model = EvadeBenchToyModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
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
    print({
        "category_accuracy": category_correct / max(total_examples, 1),
        "violation_accuracy": violation_acc / batches,
        "cross_modal_consistency_accuracy": consistency_acc / batches,
        "evasion_mask_iou": evasion_iou / batches,
        "saved_train_metrics": checkpoint.get("metrics", {}),
    })


if __name__ == "__main__":
    main()
