import argparse
import os
import time
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import LaRecDataset, collate_fn
from model import LaRecConfig, LaRecModel


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_model(checkpoint_path: str, num_items: int, device: torch.device) -> LaRecModel:
    if os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device)
        config = LaRecConfig(**payload["config"])
        model = LaRecModel(config)
        model.load_state_dict(payload["state_dict"])
    else:
        model = LaRecModel(LaRecConfig(num_items=num_items))
    return model.to(device)


def rank_target(scores: torch.Tensor, target_item_ids: torch.Tensor) -> Iterable[int]:
    target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
    return (scores > target_scores).sum(dim=1).add(1).tolist()


def hr_ndcg_at_k(ranks: Iterable[int], k: int) -> Tuple[float, float]:
    hits = []
    gains = []
    for rank in ranks:
        if rank <= k:
            hits.append(1.0)
            gains.append(1.0 / torch.log2(torch.tensor(rank + 1.0)).item())
        else:
            hits.append(0.0)
            gains.append(0.0)
    return sum(hits) / max(len(hits), 1), sum(gains) / max(len(gains), 1)


def evaluate(model: LaRecModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_ranks = []
    total_latency = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            start_time = time.perf_counter()
            scores = model.recommend(batch)
            total_latency += time.perf_counter() - start_time
            sample_count += scores.size(0)
            all_ranks.extend(rank_target(scores, batch["target_item_id"]))

    hr5, ndcg5 = hr_ndcg_at_k(all_ranks, 5)
    hr10, ndcg10 = hr_ndcg_at_k(all_ranks, 10)
    return {
        "HR@5": hr5,
        "NDCG@5": ndcg5,
        "HR@10": hr10,
        "NDCG@10": ndcg10,
        "latency_ms": (total_latency / max(sample_count, 1)) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a toy LaRec reproduction.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/larec.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = LaRecDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = load_model(args.checkpoint, test_dataset.num_items, device)

    metrics = evaluate(model, test_loader, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
