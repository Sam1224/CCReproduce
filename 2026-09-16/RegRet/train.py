from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import SyntheticRegRetConfig, SyntheticRegRetDataset
from model import RegRetConfig, RegRetToyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=4)
    parser.add_argument("--stage2-epochs", type=int, default=4)
    parser.add_argument("--stage3-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-n", type=int, default=4096)
    parser.add_argument("--val-n", type=int, default=1024)
    return parser.parse_args()


@torch.no_grad()
def recall_at_k(model: RegRetToyModel, loader: DataLoader, k: int = 1) -> float:
    model.eval()
    all_query = []
    all_text = []
    all_attr = []
    for batch in loader:
        query = model.encode_query(batch["region_feat"], batch["global_feat"])
        text = model.encode_text(batch["text_feat"])
        all_query.append(query)
        all_text.append(text)
        all_attr.append(batch["attr_id"])
    query = torch.cat(all_query, dim=0)
    text = torch.cat(all_text, dim=0)
    attr = torch.cat(all_attr, dim=0)
    sims = query @ text.t()
    topk = sims.topk(k=k, dim=1).indices
    matched_attr = attr[topk]
    return float((matched_attr == attr.unsqueeze(1)).any(dim=1).float().mean().item())


def stage1_localized_semantics(model: RegRetToyModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        logits = model.region_logits(batch["region_feat"], batch["global_feat"])
        loss = F.cross_entropy(logits, batch["attr_id"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def stage2_text_semantics(model: RegRetToyModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        logits = model.text_logits(batch["text_feat"])
        loss = F.cross_entropy(logits, batch["attr_id"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def stage3_region_contrastive(model: RegRetToyModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        query = model.encode_query(batch["region_feat"], batch["global_feat"])
        text = model.encode_text(batch["text_feat"])
        logits = (query @ text.t()) / model.cfg.temperature
        labels = torch.arange(logits.size(0))
        loss_q = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        aux_q = F.cross_entropy(model.query_head(query), batch["attr_id"])
        aux_t = F.cross_entropy(model.text_head(text), batch["attr_id"])
        loss = 0.5 * (loss_q + loss_t) + 0.2 * (aux_q + aux_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_cfg = SyntheticRegRetConfig()
    model_cfg = RegRetConfig(num_attrs=data_cfg.num_attrs)

    train_ds = SyntheticRegRetDataset(args.train_n, data_cfg, seed=args.seed)
    val_ds = SyntheticRegRetDataset(args.val_n, data_cfg, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RegRetToyModel(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "regret_toy.pt"
    best_r1 = -1.0

    for epoch in range(1, args.stage1_epochs + 1):
        loss = stage1_localized_semantics(model, train_loader, optimizer)
        r1 = recall_at_k(model, val_loader, k=1)
        print(f"stage=1 epoch={epoch} loss={loss:.4f} recall@1={r1:.4f}")

    for epoch in range(1, args.stage2_epochs + 1):
        loss = stage2_text_semantics(model, train_loader, optimizer)
        r1 = recall_at_k(model, val_loader, k=1)
        print(f"stage=2 epoch={epoch} loss={loss:.4f} recall@1={r1:.4f}")

    for epoch in range(1, args.stage3_epochs + 1):
        loss = stage3_region_contrastive(model, train_loader, optimizer)
        r1 = recall_at_k(model, val_loader, k=1)
        r5 = recall_at_k(model, val_loader, k=5)
        print(f"stage=3 epoch={epoch} loss={loss:.4f} recall@1={r1:.4f} recall@5={r5:.4f}")
        if r1 > best_r1:
            best_r1 = r1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "data_cfg": data_cfg.__dict__,
                    "model_cfg": model_cfg.__dict__,
                    "seed": args.seed,
                },
                ckpt_path,
            )

    print(f"saved: {ckpt_path} (best_recall@1={best_r1:.4f})")


if __name__ == "__main__":
    main()
