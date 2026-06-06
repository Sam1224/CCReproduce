import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class SyntheticANNData:
    base_embeddings: torch.Tensor  # [N, D]
    base_labels: torch.Tensor  # [N]
    query_embeddings: torch.Tensor  # [Q, D]
    query_labels: torch.Tensor  # [Q]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_gaussian_clusters(
    *,
    num_classes: int,
    base_per_class: int,
    query_per_class: int,
    dim: int,
    cluster_std: float,
    query_noise_std: float,
    seed: int,
) -> SyntheticANNData:
    """Generate clustered embeddings.

    We treat label agreement among retrieved neighbors as a lightweight proxy
    for downstream utility (classification / RAG topicality).
    """

    _set_seed(seed)

    centers = torch.randn(num_classes, dim)
    centers = torch.nn.functional.normalize(centers, dim=-1)

    def sample_points(per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points = []
        labels = []
        for class_id in range(num_classes):
            noise = torch.randn(per_class, dim) * cluster_std
            x = centers[class_id].unsqueeze(0) + noise
            x = torch.nn.functional.normalize(x, dim=-1)
            points.append(x)
            labels.append(torch.full((per_class,), class_id, dtype=torch.long))
        return torch.cat(points, dim=0), torch.cat(labels, dim=0)

    base_x, base_y = sample_points(base_per_class)

    # Queries are sampled near the same cluster centers, with extra noise.
    query_x, query_y = sample_points(query_per_class)
    query_x = torch.nn.functional.normalize(
        query_x + torch.randn_like(query_x) * query_noise_std, dim=-1
    )

    return SyntheticANNData(
        base_embeddings=base_x,
        base_labels=base_y,
        query_embeddings=query_x,
        query_labels=query_y,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--base_per_class", type=int, default=200)
    parser.add_argument("--query_per_class", type=int, default=40)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--cluster_std", type=float, default=0.08)
    parser.add_argument("--query_noise_std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    data = make_gaussian_clusters(
        num_classes=args.num_classes,
        base_per_class=args.base_per_class,
        query_per_class=args.query_per_class,
        dim=args.dim,
        cluster_std=args.cluster_std,
        query_noise_std=args.query_noise_std,
        seed=args.seed,
    )
    print(
        f"base={data.base_embeddings.shape}, query={data.query_embeddings.shape}, "
        f"classes={args.num_classes}"
    )


if __name__ == "__main__":
    main()
