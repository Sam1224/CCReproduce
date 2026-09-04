from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import build_datasets, set_seed
from model import COREEmbed, RankKLLoss, retrieval_metrics


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: COREEmbed, loader: DataLoader) -> dict:
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            scores = model(
                batch["query_tokens"].to(DEVICE),
                batch["candidate_features"].to(DEVICE),
            )
            all_scores.append(scores.cpu())
            all_targets.append(batch["target_index"])
    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return retrieval_metrics(scores, targets)


def main() -> None:
    set_seed(7)
    datasets = build_datasets()
    train_loader = DataLoader(datasets["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=64)

    model = COREEmbed(hidden_dim=72).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = RankKLLoss()

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, 9):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            scores = model(
                batch["query_tokens"].to(DEVICE),
                batch["candidate_features"].to(DEVICE),
            )
            teacher_scores = batch["teacher_scores"].to(DEVICE)
            loss = loss_fn(scores, teacher_scores)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss / len(train_loader), 4),
                "val_top1_acc": round(val_metrics["top1_acc"], 4),
                "val_mrr": round(val_metrics["mrr"], 4),
            }
        )
        if val_metrics["top1_acc"] > best_val:
            best_val = val_metrics["top1_acc"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    torch.save(best_state, ROOT / "core_model.pt")
    (ROOT / "train_metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_top1_acc": round(best_val, 4), "device": str(DEVICE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
