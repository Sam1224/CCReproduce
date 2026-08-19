from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SyntheticWorld:
    item_emb: torch.Tensor  # [num_items, d]
    item_category: torch.Tensor  # [num_items]
    item_price: torch.Tensor  # [num_items]
    item_quality: torch.Tensor  # [num_items]

    user_aff: torch.Tensor  # [num_users, d]
    user_rat: torch.Tensor  # [num_users, d]
    user_price_sens: torch.Tensor  # [num_users]
    user_gate_bias: torch.Tensor  # [num_users]


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def build_world(
    *,
    seed: int = 7,
    num_users: int = 1200,
    num_items: int = 800,
    d: int = 32,
    num_categories: int = 12,
) -> SyntheticWorld:
    g = torch.Generator().manual_seed(seed)

    item_emb = torch.randn(num_items, d, generator=g) / math.sqrt(d)
    item_category = torch.randint(0, num_categories, (num_items,), generator=g)
    item_price = torch.rand(num_items, generator=g)
    item_quality = torch.rand(num_items, generator=g)

    user_aff = torch.randn(num_users, d, generator=g) / math.sqrt(d)
    user_rat = torch.randn(num_users, d, generator=g) / math.sqrt(d)
    user_price_sens = torch.rand(num_users, generator=g)  # higher means more price-sensitive
    user_gate_bias = torch.randn(num_users, generator=g) * 0.5

    return SyntheticWorld(
        item_emb=item_emb,
        item_category=item_category,
        item_price=item_price,
        item_quality=item_quality,
        user_aff=user_aff,
        user_rat=user_rat,
        user_price_sens=user_price_sens,
        user_gate_bias=user_gate_bias,
    )


def _sample_candidates(
    world: SyntheticWorld,
    *,
    user_id: int,
    k: int,
    num_categories: int,
    g: torch.Generator,
) -> torch.Tensor:
    # Mix: half random, half biased toward one preferred category.
    # We keep the tensor length fixed (duplicates allowed) so batching stays simple.
    num_items = world.item_emb.shape[0]
    rand_ids = torch.randint(0, num_items, (k,), generator=g)

    # Choose a preferred category as the user's dominant category.
    pref_cat = int(torch.randint(0, num_categories, (1,), generator=g).item())
    mask = world.item_category == pref_cat
    pref_pool = torch.where(mask)[0]
    if len(pref_pool) > 0:
        pref_ids = pref_pool[torch.randint(0, len(pref_pool), (k,), generator=g)]
        cand = torch.cat([rand_ids[: k // 2], pref_ids[: k - k // 2]], dim=0)
    else:
        cand = rand_ids

    if cand.numel() < k:
        extra = torch.randint(0, num_items, (k - cand.numel(),), generator=g)
        cand = torch.cat([cand, extra], dim=0)

    return cand[:k]


def _oracle_choice(
    world: SyntheticWorld,
    *,
    user_id: int,
    cand_item_ids: torch.Tensor,
    noise_std: float,
    num_categories: int,
    g: torch.Generator,
) -> int:
    item_emb = world.item_emb[cand_item_ids]
    cat = world.item_category[cand_item_ids]
    price = world.item_price[cand_item_ids]
    quality = world.item_quality[cand_item_ids]

    u_aff = world.user_aff[user_id]
    u_rat = world.user_rat[user_id]
    price_sens = world.user_price_sens[user_id]

    # Affective: immediate attraction.
    aff = (item_emb @ u_aff) + 0.40 * quality - 0.15 * price
    # Rational: trade-off driven.
    cat_match = (cat == cat[0]).float()  # a simple (and biased) constraint token
    rat = (item_emb @ u_rat) + (1.0 - price_sens) * (quality - price) + 0.10 * cat_match

    gate = _sigmoid(world.user_gate_bias[user_id])  # scalar in (0, 1)
    score = gate * aff + (1.0 - gate) * rat

    if noise_std > 0:
        score = score + torch.randn_like(score, generator=g) * noise_std

    return int(torch.argmax(score).item())


class CaraDataset(Dataset):
    def __init__(
        self,
        world: SyntheticWorld,
        examples: List[Tuple[int, torch.Tensor, int]],
    ) -> None:
        self.world = world
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, cand, label = self.examples[idx]
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "cand_item_ids": cand.to(torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    user_ids = torch.stack([b["user_id"] for b in batch], dim=0)
    cand = torch.stack([b["cand_item_ids"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)
    return {"user_id": user_ids, "cand_item_ids": cand, "label": label}


def build_dataloaders(
    *,
    seed: int = 7,
    batch_size: int = 128,
    candidate_size: int = 48,
    num_users: int = 1200,
    num_items: int = 800,
    d: int = 32,
    num_categories: int = 12,
    train_examples: int = 18000,
    val_examples: int = 2000,
    test_examples: int = 2000,
) -> Tuple[SyntheticWorld, DataLoader, DataLoader, DataLoader]:
    world = build_world(
        seed=seed,
        num_users=num_users,
        num_items=num_items,
        d=d,
        num_categories=num_categories,
    )

    g = torch.Generator().manual_seed(seed + 1)

    def make_examples(n: int, noise_std: float) -> List[Tuple[int, torch.Tensor, int]]:
        out: List[Tuple[int, torch.Tensor, int]] = []
        for _ in range(n):
            user_id = int(torch.randint(0, num_users, (1,), generator=g).item())
            cand = _sample_candidates(
                world,
                user_id=user_id,
                k=candidate_size,
                num_categories=num_categories,
                g=g,
            )
            label = _oracle_choice(
                world,
                user_id=user_id,
                cand_item_ids=cand,
                noise_std=noise_std,
                num_categories=num_categories,
                g=g,
            )
            out.append((user_id, cand, label))
        return out

    train_ds = CaraDataset(world, make_examples(train_examples, noise_std=0.25))
    val_ds = CaraDataset(world, make_examples(val_examples, noise_std=0.0))
    test_ds = CaraDataset(world, make_examples(test_examples, noise_std=0.0))

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return world, train_dl, val_dl, test_dl


def hr_ndcg(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    # scores: [B, C]
    # labels: [B]
    max_k = max(ks)
    topk = torch.topk(scores, k=max_k, dim=-1).indices

    metrics: Dict[str, float] = {}
    bsz = scores.shape[0]
    for k in ks:
        hit = (topk[:, :k] == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
        metrics[f"hr@{k}"] = hit

        # NDCG@k (single relevant item)
        pos = torch.where(topk[:, :k] == labels.unsqueeze(-1))
        # pos[0] are batch indices, pos[1] are positions
        ndcg = torch.zeros(bsz, dtype=torch.float)
        if pos[0].numel() > 0:
            ndcg[pos[0]] = 1.0 / torch.log2(pos[1].float() + 2.0)
        metrics[f"ndcg@{k}"] = ndcg.mean().item()

    return metrics
