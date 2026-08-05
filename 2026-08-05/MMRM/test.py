from __future__ import annotations

import argparse
from typing import Dict

import torch

from data import create_dataloaders
from model import MMRM, binary_auc, ndcg_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="mmrm_toy.pt")
    parser.add_argument("--batch_size", type=int, default=128)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    seed = int(checkpoint.get("seed", 7))

    # recreate test set deterministically (the catalog used for scoring is loaded from checkpoint)
    _, _, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        seed=seed,
        num_candidates=int(config.get("num_candidates", 20)),
    )

    model = MMRM(
        vocab_size=int(config["vocab_size"]),
        image_dim=int(config["image_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_multiplex=int(config["num_multiplex"]),
        tasks=tuple(config["tasks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    catalog_text_ids = checkpoint["catalog_item_text_ids"].to(device)
    catalog_image = checkpoint["catalog_item_image"].to(device)

    metrics = evaluate(
        model,
        test_loader,
        catalog_text_ids=catalog_text_ids,
        catalog_image=catalog_image,
        device=device,
    )

    mixing = model.task_mixing_weights().detach().cpu()
    print("task->multiplex mixing weights (softmax over K):")
    for i, t in enumerate(model.tasks):
        weights = [round(float(x), 4) for x in mixing[i].tolist()]
        print(f"  {t}: {weights}")

    print(metrics)


if __name__ == "__main__":
    main()
