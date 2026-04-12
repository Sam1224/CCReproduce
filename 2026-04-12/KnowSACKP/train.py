from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ckp import build_selective_mask, compute_ckp_scores
from data import build_toy_rec_dataset
from model import MockLLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--items", type=int, default=500)
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--augment-ratio", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cpu")

    dataset = build_toy_rec_dataset(
        seed=args.seed,
        items=args.items,
        users=args.users,
        candidates=args.candidates,
    )

    llm = MockLLM(
        item_latent=dataset.item_latent,
        item_popularity=dataset.item_popularity,
        vocab_size=dataset.vocab_size,
        item_tokens=dataset.item_tokens,
    )
    llm.build_modules(device=device, dim=32)

    ks = compute_ckp_scores(dataset=dataset, llm=llm, seed=args.seed)
    mask = build_selective_mask(ks, augment_ratio=args.augment_ratio)

    os.makedirs("caches", exist_ok=True)
    torch.save(
        {
            "knowledge_score": ks,
            "augment_mask": mask,
            "meta": {
                "seed": args.seed,
                "items": args.items,
                "users": args.users,
                "candidates": args.candidates,
                "augment_ratio": args.augment_ratio,
            },
        },
        "caches/ckp_cache.pt",
    )

    print(
        f"saved caches/ckp_cache.pt | augment_ratio={args.augment_ratio:.2f} | "
        f"avg_knowledge={ks.mean():.4f} | augment_items={int(mask.sum())}"
    )


if __name__ == "__main__":
    main()
