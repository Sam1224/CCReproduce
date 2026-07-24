from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import Catalog, create_dataloaders
from model import BARGEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/barge_toy.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_catalog(saved: dict) -> Catalog:
    return Catalog(
        item_features=saved["item_features"],
        item_categories=saved["item_categories"],
        popularity=saved["popularity"],
        main_codes=saved["main_codes"],
        aux_codes=saved["aux_codes"],
        codebook_size=int(saved["codebook_size"]),
        history_len=int(saved["history_len"]),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(args.checkpoint), map_location=device)
    catalog = load_catalog(payload["catalog"])

    _, _, val_loader = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = BARGEModel(
        num_items=payload["config"]["num_items"],
        codebook_size=payload["config"]["codebook_size"],
        hidden_dim=payload["config"]["hidden_dim"],
        history_len=payload["config"]["history_len"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    plain_hits = 0
    barge_hits = 0
    dual_count = 0
    total = 0
    for batch in val_loader:
        history = batch["history"].to(device)
        target_item = batch["target_item"].to(device)
        plain = model.decode_plain(history, catalog)
        barge = model.decode_barge(history, catalog)
        plain_hits += int((plain.items == target_item).sum().item())
        barge_hits += int((barge.items == target_item).sum().item())
        dual_count += sum(1 for flag in barge.source if flag == "dual")
        total += int(target_item.numel())

    print(f"plain_recall@1={plain_hits/max(1,total):.4f}")
    print(f"barge_recall@1={barge_hits/max(1,total):.4f}")
    print(f"dual_path_ratio={dual_count/max(1,total):.4f}")


if __name__ == "__main__":
    main()
