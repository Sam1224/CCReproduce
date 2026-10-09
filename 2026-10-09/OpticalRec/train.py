from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data import (
    PreparedData,
    build_prepared_data,
    encode_queries,
    positive_ids,
    sample_negative_ids,
    stack_histories,
)
from model import ModelConfig, OpticalRecModel, bpr_loss, ndcg_at_k, recall_at_k


class ExampleDataset(Dataset):
    def __init__(
        self,
        query_tokens: torch.Tensor,
        query_masks: torch.Tensor,
        histories: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> None:
        self.query_tokens = query_tokens
        self.query_masks = query_masks
        self.histories = histories
        self.positives = positives
        self.negatives = negatives

    def __len__(self) -> int:
        return self.query_tokens.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "query_tokens": self.query_tokens[idx],
            "query_masks": self.query_masks[idx],
            "histories": self.histories[idx],
            "positives": self.positives[idx],
            "negatives": self.negatives[idx],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(prepared: PreparedData, split: str, neg_seed: int) -> ExampleDataset:
    examples = {
        "train": prepared.train_examples,
        "val": prepared.val_examples,
        "test": prepared.test_examples,
    }[split]
    q_tokens, q_masks = encode_queries(
        examples,
        vocab=prepared.vocab,
        max_query_len=prepared.max_query_len,
        pad_id=prepared.pad_id,
        unk_id=prepared.unk_id,
    )
    histories = stack_histories(examples)
    positives = positive_ids(examples)
    negatives = sample_negative_ids(examples, num_items=len(prepared.products), seed=neg_seed)
    return ExampleDataset(q_tokens, q_masks, histories, positives, negatives)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(
    model: OpticalRecModel,
    dataset: ExampleDataset,
    item_bundle: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_scores, all_pos = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            scores = model.score_items(
                query_tokens=batch["query_tokens"],
                query_masks=batch["query_masks"],
                history_ids=batch["histories"],
                item_bundle=item_bundle,
            )
            all_scores.append(scores.cpu())
            all_pos.append(batch["positives"].cpu())
    scores = torch.cat(all_scores, dim=0)
    positives = torch.cat(all_pos, dim=0)
    return {
        "recall@10": recall_at_k(scores, positives, k=10),
        "ndcg@10": ndcg_at_k(scores, positives, k=10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["optical", "late_fusion"], default="optical")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out_dir", type=str, default="runs/optical")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = build_prepared_data(seed=args.seed)
    train_ds = make_dataset(prepared, "train", neg_seed=args.seed + 11)
    val_ds = make_dataset(prepared, "val", neg_seed=args.seed + 29)
    test_ds = make_dataset(prepared, "test", neg_seed=args.seed + 53)

    cfg = ModelConfig(
        vocab_size=len(prepared.vocab),
        pad_id=prepared.pad_id,
        max_query_len=prepared.max_query_len,
        max_meta_len=prepared.max_meta_len,
    )
    model = OpticalRecModel(cfg, mode=args.mode).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    item_bundle = {
        "cards": prepared.card_tensors.to(device),
        "images": prepared.image_tensors.to(device),
        "meta_tokens": prepared.meta_tokens.to(device),
        "meta_masks": prepared.meta_masks.to(device),
    }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = to_device(batch, device)
            ctx_scores = model.score_items(
                query_tokens=batch["query_tokens"],
                query_masks=batch["query_masks"],
                history_ids=batch["histories"],
                item_bundle=item_bundle,
            )
            pos = ctx_scores.gather(1, batch["positives"].unsqueeze(1)).squeeze(1)
            neg = ctx_scores.gather(1, batch["negatives"].unsqueeze(1)).squeeze(1)

            loss = bpr_loss(pos, neg)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_ds, item_bundle, device=device, batch_size=args.batch_size)
        test_metrics = evaluate(model, test_ds, item_bundle, device=device, batch_size=args.batch_size)
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "val_recall@10": val_metrics["recall@10"],
            "val_ndcg@10": val_metrics["ndcg@10"],
            "test_recall@10": test_metrics["recall@10"],
            "test_ndcg@10": test_metrics["ndcg@10"],
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["recall@10"] > best_val:
            best_val = val_metrics["recall@10"]
            best_state = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "mode": args.mode,
                "seed": args.seed,
            }

    assert best_state is not None
    torch.save(best_state, os.path.join(args.out_dir, "model.pt"))
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "seed": args.seed,
                "num_products": len(prepared.products),
                "num_train": len(prepared.train_examples),
                "num_val": len(prepared.val_examples),
                "num_test": len(prepared.test_examples),
            },
            f,
            indent=2,
        )

    print(f"Saved best checkpoint to {os.path.join(args.out_dir, 'model.pt')}")


if __name__ == "__main__":
    main()

