from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from data import create_dataloaders
from model import MMRM, binary_auc, ndcg_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_candidates", type=int, default=20)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_multiplex", type=int, default=4)

    parser.add_argument("--purchase_weight", type=float, default=1.5)
    parser.add_argument("--checkpoint", type=str, default="mmrm_toy.pt")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: MMRM,
    loader: torch.utils.data.DataLoader,
    *,
    catalog_text_ids: torch.Tensor,
    catalog_image: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ndcg_click = 0.0
    ndcg_purchase = 0.0

    all_scores = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            batch["query_text_ids"],
            batch["history_item_ids"],
            batch["candidate_item_ids"],
            catalog_text_ids,
            catalog_image,
        )
        ndcg_click += ndcg_at_k(logits["click"], batch["click_labels"], k=10)
        ndcg_purchase += ndcg_at_k(logits["purchase"], batch["purchase_labels"], k=10)

        all_scores.append(logits["click"].detach().flatten().cpu())
        all_labels.append(batch["click_labels"].detach().flatten().cpu())

    ndcg_click /= len(loader)
    ndcg_purchase /= len(loader)
    auc_click = binary_auc(torch.cat(all_scores), torch.cat(all_labels))

    metrics = {
        "ndcg@10_click": float(ndcg_click),
        "ndcg@10_purchase": float(ndcg_purchase),
        "auc_click": float(auc_click),
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog, train_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size, seed=args.seed, num_candidates=args.num_candidates
    )

    vocab_size = int(catalog.item_text_ids.max().item()) + 1
    image_dim = int(catalog.item_image.size(-1))

    model = MMRM(
        vocab_size=vocab_size,
        image_dim=image_dim,
        hidden_dim=args.hidden_dim,
        num_multiplex=args.num_multiplex,
        tasks=("click", "purchase"),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    catalog_text_ids = catalog.item_text_ids.to(device)
    catalog_image = catalog.item_image.to(device)

    for epoch in range(args.epochs):
        model.train()
        running = {"loss": 0.0, "loss_click": 0.0, "loss_purchase": 0.0}

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                batch["query_text_ids"],
                batch["history_item_ids"],
                batch["candidate_item_ids"],
                catalog_text_ids,
                catalog_image,
            )

            loss_click = F.binary_cross_entropy_with_logits(logits["click"], batch["click_labels"])
            loss_purchase = F.binary_cross_entropy_with_logits(logits["purchase"], batch["purchase_labels"])
            loss = loss_click + args.purchase_weight * loss_purchase

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["loss_click"] += float(loss_click.item())
            running["loss_purchase"] += float(loss_purchase.item())

        running = {k: v / len(train_loader) for k, v in running.items()}
        train_metrics = evaluate(
            model,
            train_loader,
            catalog_text_ids=catalog_text_ids,
            catalog_image=catalog_image,
            device=device,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            catalog_text_ids=catalog_text_ids,
            catalog_image=catalog_image,
            device=device,
        )

        print(
            f"epoch={epoch + 1} "
            f"loss={running['loss']:.4f} click={running['loss_click']:.4f} purchase={running['loss_purchase']:.4f} "
            f"train={train_metrics} test={test_metrics}"
        )

    ckpt_path = Path(args.checkpoint)
    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "image_dim": image_dim,
            "hidden_dim": args.hidden_dim,
            "num_multiplex": args.num_multiplex,
            "tasks": ["click", "purchase"],
            "num_candidates": args.num_candidates,
        },
        "seed": args.seed,
        "catalog_item_text_ids": catalog.item_text_ids.cpu(),
        "catalog_item_image": catalog.item_image.cpu(),
    }
    torch.save(ckpt, ckpt_path)
    print(f"saved checkpoint to {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
