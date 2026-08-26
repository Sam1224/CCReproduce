from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import build_dataloaders, hr_ndcg
from model import UniSpecRec, decoupling_penalty, gather_semantic


def evaluate(model: UniSpecRec, world, loader, device: torch.device) -> dict:
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            user_id = batch["user_id"].to(device)
            cand = batch["cand_item_ids"].to(device)
            sem = gather_semantic(world, cand).to(device)
            out = model(user_id, cand, sem)
            all_scores.append(out.scores.cpu())
            all_labels.append(batch["label"].cpu())
    return hr_ndcg(torch.cat(all_scores), torch.cat(all_labels), ks=(1, 5, 10))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 26
    torch.manual_seed(seed)
    world, train_dl, val_dl, _ = build_dataloaders(seed=seed)

    model = UniSpecRec(
        num_users=world.user_cf.shape[0],
        num_items=world.item_cf.shape[0],
        d=world.item_cf.shape[1],
        sem_dim=world.semantic_smooth.shape[1],
    )
    model.init_from_world(world.item_cf, world.user_cf, world.user_sem)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "unispecrec.pt"
    history, best = [], -1.0

    for epoch in range(1, 7):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_dl:
            user_id = batch["user_id"].to(device)
            cand = batch["cand_item_ids"].to(device)
            label = batch["label"].to(device)
            sem = gather_semantic(world, cand).to(device)

            out = model(user_id, cand, sem)
            loss = F.cross_entropy(out.scores, label) + 0.03 * decoupling_penalty(out)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * user_id.shape[0]
            seen += user_id.shape[0]

        metrics = evaluate(model, world, val_dl, device)
        record = {"epoch": epoch, "train_loss": total_loss / max(seen, 1), **metrics}
        history.append(record)
        if metrics["ndcg@10"] > best:
            best = metrics["ndcg@10"]
            torch.save({"model": model.state_dict(), "seed": seed, "best_ndcg@10": best}, ckpt_path)
        print(json.dumps(record, ensure_ascii=False))

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best ndcg@10={best:.4f}; saved to {ckpt_path}")


if __name__ == "__main__":
    main()
