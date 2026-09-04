from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import build_datasets, set_seed
from model import EPICAdapter, MaskedSIDBackbone, ranking_metrics, sid_cross_entropy


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(backbone: MaskedSIDBackbone, adapter: EPICAdapter, loader: DataLoader) -> dict:
    backbone.eval()
    adapter.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            history_items = batch["history_items"].to(DEVICE)
            target_item = batch["target_item"].to(DEVICE)
            sid_logits, item_logits = backbone(history_items)
            baseline_scores = item_logits.clone()
            epic_scores = item_logits + adapter.item_posterior(history_items, [None] * 4, item_logits) * 2.5
            all_scores.append((baseline_scores.cpu(), epic_scores.cpu()))
            all_targets.append(target_item.cpu())
    baseline_scores = torch.cat([item[0] for item in all_scores], dim=0)
    epic_scores = torch.cat([item[1] for item in all_scores], dim=0)
    targets = torch.cat(all_targets, dim=0)
    baseline = ranking_metrics(baseline_scores, targets)
    epic = ranking_metrics(epic_scores, targets)
    return {
        "baseline": {key: round(value, 4) for key, value in baseline.items()},
        "epic": {key: round(value, 4) for key, value in epic.items()},
    }


def main() -> None:
    set_seed(11)
    datasets = build_datasets()
    train_loader = DataLoader(datasets["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=64)

    backbone = MaskedSIDBackbone(hidden_dim=72).to(DEVICE)
    pretrain_optim = torch.optim.AdamW(backbone.parameters(), lr=2e-3)

    for _ in range(4):
        backbone.train()
        for batch in train_loader:
            history_items = batch["history_items"].to(DEVICE)
            target_sid = batch["target_sid"].to(DEVICE)
            target_item = batch["target_item"].to(DEVICE)
            pretrain_optim.zero_grad()
            sid_logits, item_logits = backbone(history_items)
            loss = sid_cross_entropy(sid_logits, target_sid) + 0.5 * torch.nn.functional.cross_entropy(item_logits, target_item)
            loss.backward()
            pretrain_optim.step()

    for param in backbone.parameters():
        param.requires_grad = False

    adapter = EPICAdapter(hidden_dim=72).to(DEVICE)
    adapter.item_embed.weight.data.copy_(backbone.item_embed.weight.data)
    adapter_optim = torch.optim.AdamW(adapter.parameters(), lr=3e-3)
    history = []
    best_metric = -1.0
    best_adapter = None
    for epoch in range(1, 6):
        adapter.train()
        total_loss = 0.0
        for batch in train_loader:
            history_items = batch["history_items"].to(DEVICE)
            target_item = batch["target_item"].to(DEVICE)
            with torch.no_grad():
                _, item_logits = backbone(history_items)
            posterior = adapter.item_posterior(history_items, [None] * 4, item_logits)
            loss = torch.nn.functional.nll_loss(torch.log(posterior + 1e-8), target_item)
            adapter_optim.zero_grad()
            loss.backward()
            adapter_optim.step()
            total_loss += loss.item()
        val_metrics = evaluate(backbone, adapter, val_loader)
        current = val_metrics["epic"]["ndcg@5"]
        history.append({"epoch": epoch, "train_loss": round(total_loss / len(train_loader), 4), **val_metrics})
        if current > best_metric:
            best_metric = current
            best_adapter = {key: value.cpu() for key, value in adapter.state_dict().items()}

    torch.save({"backbone": backbone.state_dict(), "adapter": best_adapter}, ROOT / "epic_model.pt")
    (ROOT / "train_metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_ndcg@5": round(best_metric, 4), "device": str(DEVICE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
