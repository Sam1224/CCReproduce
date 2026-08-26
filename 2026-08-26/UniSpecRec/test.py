from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_dataloaders, hr_ndcg
from model import UniSpecRec, gather_semantic


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 26
    world, _, _, test_dl = build_dataloaders(seed=seed)

    model = UniSpecRec(
        num_users=world.user_cf.shape[0],
        num_items=world.item_cf.shape[0],
        d=world.item_cf.shape[1],
        sem_dim=world.semantic_smooth.shape[1],
    )
    model.init_from_world(world.item_cf, world.user_cf, world.user_sem)

    ckpt = Path(__file__).resolve().parent / "artifacts" / "unispecrec.pt"
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            user_id = batch["user_id"].to(device)
            cand = batch["cand_item_ids"].to(device)
            sem = gather_semantic(world, cand).to(device)
            out = model(user_id, cand, sem)
            all_scores.append(out.scores.cpu())
            all_labels.append(batch["label"].cpu())

    metrics = hr_ndcg(torch.cat(all_scores), torch.cat(all_labels), ks=(1, 5, 10))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
