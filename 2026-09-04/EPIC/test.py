from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_datasets
from model import EPICAdapter, MaskedSIDBackbone, decode_with_epic, rank_items_from_sid, ranking_metrics


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    datasets = build_datasets()
    test_loader = DataLoader(datasets["test"], batch_size=64)

    payload = torch.load(ROOT / "epic_model.pt", map_location=DEVICE)
    backbone = MaskedSIDBackbone(hidden_dim=72).to(DEVICE)
    adapter = EPICAdapter(hidden_dim=72).to(DEVICE)
    backbone.load_state_dict(payload["backbone"])
    adapter.load_state_dict(payload["adapter"])
    backbone.eval()
    adapter.eval()

    baseline_scores = []
    epic_scores = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            history_items = batch["history_items"].to(DEVICE)
            target_item = batch["target_item"].to(DEVICE)
            _, item_logits = backbone(history_items)
            epic_sid = decode_with_epic(backbone, adapter, history_items, use_epic=True)
            baseline_scores.append(item_logits.cpu())
            epic_scores.append(rank_items_from_sid(epic_sid, item_logits, adapter=adapter, history_items=history_items).cpu())
            all_targets.append(target_item.cpu())

    targets = torch.cat(all_targets, dim=0)
    baseline = ranking_metrics(torch.cat(baseline_scores, dim=0), targets)
    epic = ranking_metrics(torch.cat(epic_scores, dim=0), targets)
    metrics = {
        "baseline": baseline,
        "epic": epic,
        "delta_ndcg@5": round(epic["ndcg@5"] - baseline["ndcg@5"], 4),
        "delta_recall@5": round(epic["recall@5"] - baseline["recall@5"], 4),
    }
    (ROOT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
