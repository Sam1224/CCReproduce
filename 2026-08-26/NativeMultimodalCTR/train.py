from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from data import binary_metrics, build_dataloaders, make_batch, make_triplet_batch
from model import NativeMultimodalCTR


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(model: NativeMultimodalCTR, world, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for raw in loader:
            batch = to_device(make_batch(world, raw), device)
            out = model(
                user_id=batch["user_id"],
                text=batch["text"],
                image=batch["image"],
                category=batch["category"],
                price=batch["price"],
            )
            logits.append(out.logits.cpu())
            labels.append(batch["label"].cpu())
    return binary_metrics(torch.cat(logits), torch.cat(labels))


def main() -> None:
    seed = 26
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world, triplet_dl, train_dl, val_dl, _ = build_dataloaders(seed=seed)
    model = NativeMultimodalCTR(
        num_users=world.user_pref.shape[0],
        num_categories=int(world.category.max()) + 1,
        text_dim=world.text_feat.shape[1],
        image_dim=world.image_feat.shape[1],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "native_multimodal_ctr.pt"
    history = []

    # Stage 1: mine-then-train multimodal encoder with triplet loss.
    for epoch in range(1, 4):
        model.train()
        total = 0.0
        seen = 0
        for raw in triplet_dl:
            batch = to_device(make_triplet_batch(world, raw), device)
            loss = model.triplet_loss(**batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * raw["anchor"].shape[0]
            seen += raw["anchor"].shape[0]
        rec = {"stage": "triplet", "epoch": epoch, "loss": total / max(seen, 1)}
        history.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    best_auc = -1.0
    # Stage 2: CTR fine-tuning with the pretrained native multimodal encoder.
    for epoch in range(1, 7):
        model.train()
        total = 0.0
        seen = 0
        for raw in train_dl:
            batch = to_device(make_batch(world, raw), device)
            out = model(
                user_id=batch["user_id"],
                text=batch["text"],
                image=batch["image"],
                category=batch["category"],
                price=batch["price"],
            )
            loss = F.binary_cross_entropy_with_logits(out.logits, batch["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * batch["label"].shape[0]
            seen += batch["label"].shape[0]

        metrics = evaluate(model, world, val_dl, device)
        rec = {"stage": "ctr", "epoch": epoch, "loss": total / max(seen, 1), **metrics}
        history.append(rec)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save({"model": model.state_dict(), "seed": seed}, ckpt_path)
        print(json.dumps(rec, ensure_ascii=False))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best val auc={best_auc:.4f}; saved to {ckpt_path}")


if __name__ == "__main__":
    main()
