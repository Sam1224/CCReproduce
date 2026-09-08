from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import build_datasets, set_seed
from model import RewardAlignmentLoss, SAMD2QModel, token_recall_at_k

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: SAMD2QModel, loader: DataLoader) -> dict:
    model.eval()
    all_logits = []
    all_masks = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["title_ids"].to(DEVICE),
                batch["image_features"].to(DEVICE),
            )
            all_logits.append(logits.cpu())
            all_masks.append(batch["positive_mask"])
    logits = torch.cat(all_logits, dim=0)
    masks = torch.cat(all_masks, dim=0)
    return {
        "token_recall_at_2": round(token_recall_at_k(logits, masks, k=2), 4),
        "token_recall_at_3": round(token_recall_at_k(logits, masks, k=3), 4),
    }


def main() -> None:
    set_seed(7)
    datasets = build_datasets()
    train_loader = DataLoader(datasets["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=64)

    model = SAMD2QModel(hidden_dim=80).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = RewardAlignmentLoss()

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, 11):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["title_ids"].to(DEVICE),
                batch["image_features"].to(DEVICE),
            )
            reward_targets = batch["reward_targets"].to(DEVICE)
            loss = loss_fn(logits, reward_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss / len(train_loader), 4),
                **val_metrics,
            }
        )
        if val_metrics["token_recall_at_2"] > best_val:
            best_val = val_metrics["token_recall_at_2"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    torch.save(best_state, ROOT / "sam_d2q.pt")
    (ROOT / "train_metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_val_token_recall_at_2": round(best_val, 4), "device": str(DEVICE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
