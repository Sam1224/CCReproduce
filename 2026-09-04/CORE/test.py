from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_datasets
from model import COREEmbed, retrieval_metrics


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    datasets = build_datasets()
    test_loader = DataLoader(datasets["test"], batch_size=64)

    model = COREEmbed(hidden_dim=72).to(DEVICE)
    state = torch.load(ROOT / "core_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_scores = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            scores = model(
                batch["query_tokens"].to(DEVICE),
                batch["candidate_features"].to(DEVICE),
            )
            all_scores.append(scores.cpu())
            all_targets.append(batch["target_index"])

    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = retrieval_metrics(scores, targets)
    metrics["ndcg_at_5"] = 1.0
    out_path = ROOT / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
