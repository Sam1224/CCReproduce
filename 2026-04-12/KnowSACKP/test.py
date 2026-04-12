from __future__ import annotations

import argparse

import numpy as np
import torch

from data import build_toy_rec_dataset
from model import MockLLM


def recall_at_1(scores: np.ndarray, candidates: list[int], gt: int) -> float:
    best = candidates[int(np.argmax(scores))]
    return 1.0 if best == gt else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="caches/ckp_cache.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    meta = cache["meta"]

    dataset = build_toy_rec_dataset(
        seed=meta["seed"],
        items=meta["items"],
        users=meta["users"],
        candidates=meta["candidates"],
    )

    llm = MockLLM(
        item_latent=dataset.item_latent,
        item_popularity=dataset.item_popularity,
        vocab_size=dataset.vocab_size,
        item_tokens=dataset.item_tokens,
    )
    llm.build_modules(device=device, dim=32)

    augment_mask = torch.tensor(cache["augment_mask"], dtype=torch.bool, device=device)

    policies = ["no_augment", "uniform", "selective"]
    results = {p: [] for p in policies}

    for case in dataset.user_cases:
        hist_t = torch.tensor(case.history, dtype=torch.long, device=device)
        with torch.no_grad():
            user_vec = llm.param_emb[hist_t].mean(dim=0)
            user_vec = user_vec / (user_vec.norm() + 1e-12)

        cand_ids = torch.tensor(case.candidates, dtype=torch.long, device=device)
        for policy in policies:
            item_repr = llm.get_item_repr(policy, cand_ids, augment_mask=augment_mask)
            with torch.no_grad():
                s = llm.score_items(user_vec, cand_ids, item_repr).detach().cpu().numpy()
            results[policy].append(recall_at_1(s, case.candidates, case.ground_truth))

    print("=== Recall@1 (toy) ===")
    for policy in policies:
        print(f"{policy:10s}  {float(np.mean(results[policy])):.4f}")


if __name__ == "__main__":
    main()
