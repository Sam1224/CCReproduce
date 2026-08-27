from __future__ import annotations

import torch

from data import build_continual_loaders
from model import VDAModel


def main() -> None:
    _, test_loaders = build_continual_loaders()
    model = VDAModel()
    model.load_state_dict(torch.load("vda_toy.pt", map_location="cpu"))
    model.eval()
    accs = []
    with torch.no_grad():
        for idx, loader in enumerate(test_loaders):
            correct = total = 0
            for batch in loader:
                pred = model(batch).logits.argmax(dim=-1)
                correct += int((pred == batch["labels"]).sum())
                total += int(batch["labels"].numel())
            acc = correct / max(total, 1)
            accs.append(acc)
            print(f"task_{idx}_accuracy={acc:.3f}")
    print(f"avg_accuracy={sum(accs) / len(accs):.3f}")


if __name__ == "__main__":
    main()
