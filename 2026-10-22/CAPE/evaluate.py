from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cape import CapeConfig, cape_rank
from dataset import ToySequentialDataset, build_cooccurrence, split_dataset
from model import SemanticIdRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/backbone.pt"))
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--expansion-size", type=int, default=25)
    parser.add_argument("--lambda-mem", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def hit_rate(ranked: torch.Tensor, target: torch.Tensor) -> float:
    return (ranked == target[:, None]).any(dim=1).float().mean().item()


def ndcg(ranked: torch.Tensor, target: torch.Tensor) -> float:
    matches = ranked == target[:, None]
    gains = torch.zeros(ranked.size(0), device=ranked.device)
    rows = matches.any(dim=1)
    if rows.any():
        ranks = matches.float().argmax(dim=1)[rows].float() + 1.0
        gains[rows] = 1.0 / torch.log2(ranks + 1.0)
    return gains.mean().item()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    dataset = ToySequentialDataset()
    train_examples, test_examples = split_dataset(dataset)
    memory = build_cooccurrence(train_examples, dataset.num_items, device=args.device)

    model = SemanticIdRecommender(dataset.num_items).to(args.device)
    if args.checkpoint.exists():
        payload = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(payload["model"])
    else:
        print("checkpoint missing; evaluating randomly initialized backbone")
    model.eval()

    history = torch.tensor([row.history for row in test_examples], device=args.device)
    target = torch.tensor([row.target for row in test_examples], device=args.device)
    tail_mask = target >= dataset.head_cutoff

    _, backbone = model.beam_candidates(history, args.top_k)
    config = CapeConfig(
        beam_size=args.beam_size,
        expansion_size=args.expansion_size,
        top_k=args.top_k,
        lambda_mem=args.lambda_mem,
        head_cutoff=dataset.head_cutoff,
    )
    cape, _ = cape_rank(model, history, memory, config)

    for name, ranked in [("backbone", backbone), ("cape", cape)]:
        print(
            f"{name}: HR@{args.top_k}={hit_rate(ranked, target):.4f} "
            f"NDCG@{args.top_k}={ndcg(ranked, target):.4f} "
            f"tail_HR@{args.top_k}={hit_rate(ranked[tail_mask], target[tail_mask]):.4f} "
            f"tail_NDCG@{args.top_k}={ndcg(ranked[tail_mask], target[tail_mask]):.4f}"
        )


if __name__ == "__main__":
    main()

