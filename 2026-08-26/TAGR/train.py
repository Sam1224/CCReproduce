from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_dataloaders, hr_ndcg
from model import TAGRModel, gather_item_side, tagr_loss


def evaluate(model: TAGRModel, world, loader, device: torch.device) -> dict:
    model.eval()
    all_scores, all_labels, all_rewards = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            cat, price, quality = gather_item_side(world, batch["cand_item_ids"].cpu())
            out = model(
                user_id=batch["user_id"],
                room_id=batch["room_id"],
                hist_item_ids=batch["hist_item_ids"],
                hist_actions=batch["hist_actions"],
                cand_item_ids=batch["cand_item_ids"],
                stage=batch["stage"],
                focus_cat=batch["focus_cat"],
                promo=batch["promo"],
                freshness=batch["freshness"],
                item_category=cat.to(device),
                item_price=price.to(device),
                item_quality=quality.to(device),
            )
            all_scores.append(out.scores.cpu())
            all_labels.append(batch["label"].cpu())
            all_rewards.append(batch["reward_proxy"].cpu())
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    rewards = torch.cat(all_rewards, dim=0)
    metrics = hr_ndcg(scores, labels, ks=(1, 5, 10))
    pred = scores.argmax(dim=-1)
    metrics["avg_proxy_reward"] = rewards.gather(1, pred[:, None]).mean().item()
    return metrics


def main() -> None:
    seed = 26
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, train_dl, val_dl, _ = build_dataloaders(seed=seed)

    model = TAGRModel(
        num_users=world.user_pref.shape[0],
        num_ads=world.ad_emb.shape[0],
        num_rooms=world.room_style.shape[0],
        num_categories=int(world.ad_category.max().item()) + 1,
        d=world.ad_emb.shape[1],
    )
    model.init_from_world(user_pref=world.user_pref, ad_emb=world.ad_emb, room_style=world.room_style)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "tagr.pt"
    history = []
    best = -1.0

    for epoch in range(1, 6):
        model.train()
        total, seen = 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            cat, price, quality = gather_item_side(world, batch["cand_item_ids"].cpu())
            out = model(
                user_id=batch["user_id"],
                room_id=batch["room_id"],
                hist_item_ids=batch["hist_item_ids"],
                hist_actions=batch["hist_actions"],
                cand_item_ids=batch["cand_item_ids"],
                stage=batch["stage"],
                focus_cat=batch["focus_cat"],
                promo=batch["promo"],
                freshness=batch["freshness"],
                item_category=cat.to(device),
                item_price=price.to(device),
                item_quality=quality.to(device),
            )
            loss = tagr_loss(out.scores, batch["label"], batch["reward_proxy"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * batch["label"].shape[0]
            seen += batch["label"].shape[0]

        metrics = evaluate(model, world, val_dl, device)
        record = {"epoch": epoch, "train_loss": total / max(seen, 1), **metrics}
        history.append(record)
        if metrics["ndcg@10"] > best:
            best = metrics["ndcg@10"]
            torch.save({"model": model.state_dict(), "seed": seed}, ckpt_path)
        print(json.dumps(record, ensure_ascii=False))

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best ndcg@10={best:.4f}; saved to {ckpt_path}")


if __name__ == "__main__":
    main()
