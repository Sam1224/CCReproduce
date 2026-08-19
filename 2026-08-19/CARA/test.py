from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_dataloaders, hr_ndcg
from model import CaraModel, gather_item_side


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 7
    world, _, _, test_dl = build_dataloaders(seed=seed)

    model = CaraModel(
        num_users=world.user_aff.shape[0],
        num_items=world.item_emb.shape[0],
        num_categories=int(world.item_category.max().item()) + 1,
        d=world.item_emb.shape[1],
    )
    model.init_from_world(item_emb=world.item_emb, user_aff=world.user_aff, user_rat=world.user_rat)

    ckpt = Path(__file__).resolve().parent / "artifacts" / "cara.pt"
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dl:
            user_id = batch["user_id"].to(device)
            cand_item_ids = batch["cand_item_ids"].to(device)
            label = batch["label"].to(device)

            cat, price, quality = gather_item_side(world, cand_item_ids)
            out = model(
                user_id=user_id,
                cand_item_ids=cand_item_ids,
                item_category=cat.to(device),
                item_price=price.to(device),
                item_quality=quality.to(device),
                filter_topk=24,
            )
            all_scores.append(out.scores.cpu())
            all_labels.append(label.cpu())

    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = hr_ndcg(scores, labels, ks=(1, 5, 10))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
