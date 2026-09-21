from __future__ import annotations

import json
from pathlib import Path

import torch

from data import SelectivePredictionDataset
from model import ConfidenceGate, best_contiguous_partition, certificate_coverage


def main() -> None:
    dataset = SelectivePredictionDataset(size=240, seed=47)
    gate = ConfidenceGate()
    gate.load_state_dict(torch.load("artifacts/confidence_gate.pt", map_location="cpu"))
    gate.eval()
    with torch.no_grad():
        prob = gate(dataset.features).softmax(dim=-1)
        confidence, pred = prob.max(dim=-1)
        correct = pred.eq(dataset.label)
        coverage, upper_error = certificate_coverage(confidence, correct, threshold=0.70, target_precision=0.85)
        qualities = [float(correct[dataset.group == i].float().mean().item()) for i in range(12)]
        segments = best_contiguous_partition(qualities, 4)
    print(f"certified coverage={coverage:.4f} upper_error={upper_error:.4f} segments={len(segments)}")
    assert len(segments) == 4
    assert Path("artifacts/metrics.json").exists()


if __name__ == "__main__":
    main()
