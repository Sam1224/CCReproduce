from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_dataloaders, hr_ndcg
from model import TAGRModel, gather_item_side


def main() -> None:
    seed = 26
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world, _, _, test_dl = build_dataloaders(seed=seed)
    model = TAGRModel(
        num_users=world.user_pref.shape[0],
        num_ads=world.ad_emb.shape[0],
        num_rooms=world.room_style.shape[0],
        num_categories=int(world.ad_category.max().item()) + 1,
        d=world.ad_emb.shape[1],
    )
    model.init_from_world(user_pref=world.user_pref, ad_emb=world.ad_emb, room_style=world.room_style)

    ckpt = Path(__file__).resolve().parent / "artifacts" / "tagr.pt"
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    all_scores, all_labels, all_rewards = [], [], []
    with torch.no_grad():
        for batch in test_dl:
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
    pred = scores.argmax(dim=-1)
    metrics = hr_ndcg(scores, labels, ks=(1, 5, 10))
    metrics["avg_proxy_reward"] = rewards.gather(1, pred[:, None]).mean().item()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
