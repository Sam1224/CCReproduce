from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset

from dataset import CompositionalListDataset, collate_samples, split_by_group
from model import COREConfig, COREEmbeddingModel


def ndcg(levels: torch.Tensor, scores: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    gains = torch.pow(2.0, levels.float()[order]) - 1.0
    discounts = torch.log2(torch.arange(2, 2 + gains.numel(), device=levels.device).float())
    dcg = torch.sum(gains / discounts)
    ideal = torch.sort(levels.float(), descending=True).values
    ideal_gains = torch.pow(2.0, ideal) - 1.0
    ideal_dcg = torch.sum(ideal_gains / discounts)
    return float((dcg / ideal_dcg.clamp(min=1e-8)).item())


@torch.no_grad()
def evaluate_rankings(model: COREEmbeddingModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    grouped: Dict[int, List[tuple[float, int]]] = {}
    for batch in loader:
        query_tokens = batch["query_tokens"].to(device)
        candidate_tokens = batch["candidate_tokens"].to(device)
        image_features = batch["image_features"].to(device)
        scores = model(query_tokens, candidate_tokens, image_features).cpu()
        for score, level, group_id in zip(scores, batch["level"], batch["group_id"]):
            grouped.setdefault(int(group_id.item()), []).append((float(score.item()), int(level.item())))

    recall_hits = 0
    ndcg_values = []
    pair_correct = 0
    pair_total = 0
    for rows in grouped.values():
        score_tensor = torch.tensor([row[0] for row in rows])
        level_tensor = torch.tensor([row[1] for row in rows])
        top_idx = int(torch.argmax(score_tensor).item())
        recall_hits += int(level_tensor[top_idx].item() == 5)
        ndcg_values.append(ndcg(level_tensor, score_tensor))
        for i in range(len(rows)):
            for j in range(len(rows)):
                if level_tensor[i] > level_tensor[j]:
                    pair_correct += int(score_tensor[i] > score_tensor[j])
                    pair_total += 1

    group_count = max(len(grouped), 1)
    return {
        "recall_at_1": recall_hits / group_count,
        "ndcg_at_5": sum(ndcg_values) / group_count,
        "pairwise_order_acc": pair_correct / max(pair_total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/core_embed.pt")
    parser.add_argument("--num_groups", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    dataset = CompositionalListDataset(num_groups=args.num_groups)
    _, test_idx = split_by_group(dataset)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location=device)
    cfg = COREConfig(**payload["config"])
    model = COREEmbeddingModel(cfg).to(device)
    model.load_state_dict(payload["model"])
    metrics = evaluate_rankings(model, loader, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
