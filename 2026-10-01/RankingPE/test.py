from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_splits, set_seed
from model import ToyMultimodalJudge
from train import DEVICE, evaluate_prompt, select_prompts

ROOT = Path(__file__).resolve().parent


def main() -> None:
    set_seed(29)
    splits = build_splits(seed=13)
    model = ToyMultimodalJudge(hidden_dim=48).to(DEVICE)
    checkpoint = torch.load(ROOT / "ranking_pe_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    valid_features = splits.valid.features.to(DEVICE)
    valid_labels = splits.valid.labels.to(DEVICE)
    selection = select_prompts(model, valid_features, valid_labels)

    test_features = splits.test.features.to(DEVICE)
    test_labels = splits.test.labels.to(DEVICE)
    metrics = {
        "accuracy_pe": evaluate_prompt(model, test_features, test_labels, selection["accuracy_pe"]["prompt"]),
        "ranking_pe": evaluate_prompt(model, test_features, test_labels, selection["ranking_pe"]["prompt"]),
    }
    improvement = round(metrics["ranking_pe"]["auroc"] - metrics["accuracy_pe"]["auroc"], 4)
    output = {"metrics": metrics, "auroc_improvement": improvement, "selection": selection}
    (ROOT / "test_metrics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
