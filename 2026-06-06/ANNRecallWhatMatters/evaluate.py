import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from model import RandomProjectionANNIndex, exact_knn


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="outputs/run1")
    parser.add_argument("--k", type=int, default=20)

    # Sweep controls the approximation strength.
    parser.add_argument(
        "--proj_dims",
        type=str,
        default="4,8,16,32,64",
        help="comma-separated projection dims to evaluate",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def recall_at_k(approx_idx: torch.Tensor, exact_idx: torch.Tensor) -> float:
    # average per-query overlap ratio
    exact_sets = [set(row.tolist()) for row in exact_idx]
    overlaps = []
    for q, row in enumerate(approx_idx):
        hit = sum(1 for x in row.tolist() if x in exact_sets[q])
        overlaps.append(hit / len(row))
    return float(np.mean(overlaps))


def inv_ratio_at_k(approx_dist: torch.Tensor, exact_dist: torch.Tensor) -> float:
    # 1/Ratio@k = mean_i (d*_i / d~_i), bounded by (0, 1]
    eps = 1e-12
    ratio = exact_dist / torch.clamp(approx_dist, min=eps)
    ratio = torch.clamp(ratio, max=1.0)
    return float(ratio.mean().item())


def label_precision_at_k(
    approx_idx: torch.Tensor, base_labels: torch.Tensor, query_labels: torch.Tensor
) -> float:
    labels = base_labels[approx_idx]  # [Q, k]
    correct = labels.eq(query_labels.unsqueeze(1)).float()
    return float(correct.mean().item())


def main() -> None:
    args = build_argparser().parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu")

    base_embeddings = ckpt["base_embeddings"].float()
    base_labels = ckpt["base_labels"].long()
    query_embeddings = ckpt["query_embeddings"].float()
    query_labels = ckpt["query_labels"].long()

    exact_idx, exact_dist = exact_knn(base_embeddings, query_embeddings, k=args.k)

    proj_dims = [int(x) for x in args.proj_dims.split(",") if x.strip()]

    rows: List[Dict[str, float]] = []
    for proj_dim in proj_dims:
        index = RandomProjectionANNIndex.build(
            base_embeddings=base_embeddings,
            base_labels=base_labels,
            proj_dim=proj_dim,
            seed=args.seed,
        )
        approx_idx, approx_dist = index.search(query_embeddings, k=args.k)

        rows.append(
            {
                "proj_dim": float(proj_dim),
                "recall@k": recall_at_k(approx_idx, exact_idx),
                "inv_ratio@k": inv_ratio_at_k(approx_dist, exact_dist),
                "label_precision@k": label_precision_at_k(
                    approx_idx, base_labels, query_labels
                ),
            }
        )

    out = {"k": args.k, "results": rows}
    out_path = run_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
