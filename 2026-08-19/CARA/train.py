from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import build_dataloaders, hr_ndcg
from model import CaraModel, boundary_weight, gather_item_side


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 7
    torch.manual_seed(seed)

    world, train_dl, val_dl, _ = build_dataloaders(seed=seed)

    model = CaraModel(
        num_users=world.user_aff.shape[0],
        num_items=world.item_emb.shape[0],
        num_categories=int(world.item_category.max().item()) + 1,
        d=world.item_emb.shape[1],
    )
    model.init_from_world(item_emb=world.item_emb, user_aff=world.user_aff, user_rat=world.user_rat)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "cara.pt"

    best = -1.0
    history = []

    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        seen = 0

        for batch in train_dl:
            user_id = batch["user_id"].to(device)
            cand_item_ids = batch["cand_item_ids"].to(device)
            label = batch["label"].to(device)

            cat, price, quality = gather_item_side(world, cand_item_ids)
            cat = cat.to(device)
            price = price.to(device)
            quality = quality.to(device)

            out = model(
                user_id=user_id,
                cand_item_ids=cand_item_ids,
                item_category=cat,
                item_price=price,
                item_quality=quality,
                filter_topk=None,
            )

            ce = F.cross_entropy(out.scores, label, reduction="none")

            with torch.no_grad():
                p = torch.softmax(out.scores, dim=-1)
                p_correct = p.gather(1, label.unsqueeze(1)).squeeze(1)
                w = boundary_weight(p_correct)

            loss = (w * ce).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.item()) * user_id.shape[0]
            seen += user_id.shape[0]

        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dl:
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
                    filter_topk=None,
                )
                all_scores.append(out.scores.cpu())
                all_labels.append(label.cpu())

        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        m = hr_ndcg(scores, labels, ks=(1, 5, 10))
        metric = m["ndcg@10"]

        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            **m,
        }
        history.append(record)

        if metric > best:
            best = metric
            torch.save({"model": model.state_dict()}, ckpt_path)

        print(json.dumps(record, ensure_ascii=False))

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best ndcg@10={best:.4f}; saved to {ckpt_path}")


if __name__ == "__main__":
    main()
