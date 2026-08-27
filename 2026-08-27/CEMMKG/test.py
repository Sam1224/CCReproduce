from __future__ import annotations

import torch

from data import VOCAB, build_dataloaders
from model import CEMMKGModel


def main() -> None:
    _, _, test_loader = build_dataloaders()
    model = CEMMKGModel(vocab_size=len(VOCAB))
    state = torch.load("cemmkg_toy.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    correct = total = 0
    gates = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            correct += int((out.logits.argmax(dim=-1) == batch["label"]).sum())
            total += int(batch["label"].numel())
            gates.append(out.context_gate.mean(dim=0))
    gate = torch.stack(gates).mean(dim=0)
    print(f"strict_accuracy={correct / max(total, 1):.3f}")
    print(f"avg_context_gate visual/local/global={gate.tolist()}")


if __name__ == "__main__":
    main()
