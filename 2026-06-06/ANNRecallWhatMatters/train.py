import argparse
import json
import os
from pathlib import Path

import torch

from data import make_gaussian_clusters
from model import RandomProjectionANNIndex


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs/run1")

    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--base_per_class", type=int, default=200)
    parser.add_argument("--query_per_class", type=int, default=40)
    parser.add_argument("--dim", type=int, default=256)

    parser.add_argument("--proj_dim", type=int, default=32)
    parser.add_argument("--cluster_std", type=float, default=0.08)
    parser.add_argument("--query_noise_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = make_gaussian_clusters(
        num_classes=args.num_classes,
        base_per_class=args.base_per_class,
        query_per_class=args.query_per_class,
        dim=args.dim,
        cluster_std=args.cluster_std,
        query_noise_std=args.query_noise_std,
        seed=args.seed,
    )

    index = RandomProjectionANNIndex.build(
        base_embeddings=data.base_embeddings,
        base_labels=data.base_labels,
        proj_dim=args.proj_dim,
        seed=args.seed,
    )

    torch.save(
        {
            "config": vars(args),
            "base_embeddings": data.base_embeddings,
            "base_labels": data.base_labels,
            "query_embeddings": data.query_embeddings,
            "query_labels": data.query_labels,
            "proj": index.proj,
        },
        out_dir / "checkpoint.pt",
    )

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
