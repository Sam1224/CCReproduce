from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import SelectivePredictionDataset
from model import ConfidenceGate, best_contiguous_partition, certificate_coverage


def main() -> None:
    torch.manual_seed(43)
    dataset = SelectivePredictionDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    gate = ConfidenceGate()
    opt = torch.optim.AdamW(gate.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(5):
        total = 0.0
        for batch in loader:
            logits = gate(batch["x"])
            loss = loss_fn(logits, batch["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total / len(loader):.4f}")
    with torch.no_grad():
        all_x = dataset.features
        pred = gate(all_x).softmax(dim=-1)
        confidence, label = pred.max(dim=-1)
        correct = label.eq(dataset.label)
        coverage, upper_error = certificate_coverage(confidence, correct, threshold=0.72, target_precision=0.9)
        qualities = []
        for group in range(12):
            mask = dataset.group == group
            qualities.append(float(correct[mask].float().mean().item()))
        segments = [segment.__dict__ for segment in best_contiguous_partition(qualities, 4)]
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(gate.state_dict(), "artifacts/confidence_gate.pt")
    Path("artifacts/metrics.json").write_text(json.dumps({"coverage": coverage, "upper_error": upper_error, "segments": segments}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
