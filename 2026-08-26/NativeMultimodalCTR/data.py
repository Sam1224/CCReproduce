from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SyntheticWorld:
    text_feat: torch.Tensor
    image_feat: torch.Tensor
    category: torch.Tensor
    price: torch.Tensor
    true_item: torch.Tensor
    user_pref: torch.Tensor
    user_price_sens: torch.Tensor
    user_cat_pref: torch.Tensor


LogRow = Tuple[int, int, int]  # user_id, item_id, clicked
TripletRow = Tuple[int, int, int]  # anchor item, positive item, negative item


def build_world(
    seed: int = 26,
    num_users: int = 700,
    num_items: int = 500,
    text_dim: int = 32,
    image_dim: int = 32,
    latent_dim: int = 24,
    num_categories: int = 10,
) -> SyntheticWorld:
    g = torch.Generator().manual_seed(seed)
    text_feat = torch.randn(num_items, text_dim, generator=g)
    image_feat = torch.randn(num_items, image_dim, generator=g)
    category = torch.randint(0, num_categories, (num_items,), generator=g)
    price = torch.rand(num_items, generator=g)

    w_text = torch.randn(text_dim, latent_dim, generator=g) / math.sqrt(text_dim)
    w_image = torch.randn(image_dim, latent_dim, generator=g) / math.sqrt(image_dim)
    cat_proto = torch.randn(num_categories, latent_dim, generator=g) * 0.25
    true_item = torch.tanh(text_feat @ w_text + image_feat @ w_image + cat_proto[category])
    true_item = torch.nn.functional.normalize(true_item, dim=-1)

    user_pref = torch.nn.functional.normalize(torch.randn(num_users, latent_dim, generator=g), dim=-1)
    user_price_sens = torch.rand(num_users, generator=g)
    user_cat_pref = torch.randint(0, num_categories, (num_users,), generator=g)
    return SyntheticWorld(text_feat, image_feat, category, price, true_item, user_pref, user_price_sens, user_cat_pref)


def _click_logit(world: SyntheticWorld, user_id: int, item_id: int) -> float:
    affinity = float((world.user_pref[user_id] * world.true_item[item_id]).sum())
    cat_bonus = 0.45 if int(world.category[item_id]) == int(world.user_cat_pref[user_id]) else -0.10
    price_penalty = float(world.user_price_sens[user_id] * world.price[item_id])
    return 3.0 * affinity + cat_bonus - 0.9 * price_penalty


def build_logs(world: SyntheticWorld, seed: int = 26, n: int = 12000) -> List[LogRow]:
    g = torch.Generator().manual_seed(seed + 1)
    rows: List[LogRow] = []
    num_users = world.user_pref.shape[0]
    num_items = world.text_feat.shape[0]
    for _ in range(n):
        user_id = int(torch.randint(0, num_users, (1,), generator=g))
        if torch.rand((), generator=g).item() < 0.45:
            pool = torch.where(world.category == world.user_cat_pref[user_id])[0]
            item_id = int(pool[torch.randint(0, len(pool), (1,), generator=g)])
        else:
            item_id = int(torch.randint(0, num_items, (1,), generator=g))
        prob = torch.sigmoid(torch.tensor(_click_logit(world, user_id, item_id))).item()
        clicked = int(torch.rand((), generator=g).item() < prob)
        rows.append((user_id, item_id, clicked))
    return rows


def split_logs(rows: Sequence[LogRow]) -> Tuple[List[LogRow], List[LogRow], List[LogRow]]:
    n = len(rows)
    return list(rows[: int(0.70 * n)]), list(rows[int(0.70 * n) : int(0.85 * n)]), list(rows[int(0.85 * n) :])


def mine_triplets(rows: Sequence[LogRow], world: SyntheticWorld, seed: int = 26, max_triplets: int = 8000) -> List[TripletRow]:
    """Mine clicked anchor/positive pairs and hard skipped negatives from behavior logs."""

    rng = random.Random(seed + 2)
    by_user_pos: Dict[int, List[int]] = {}
    by_user_neg: Dict[int, List[int]] = {}
    for user_id, item_id, clicked in rows:
        (by_user_pos if clicked else by_user_neg).setdefault(user_id, []).append(item_id)

    triplets: List[TripletRow] = []
    users = list(by_user_pos.keys())
    rng.shuffle(users)
    for user_id in users:
        pos = list(dict.fromkeys(by_user_pos.get(user_id, [])))
        neg = list(dict.fromkeys(by_user_neg.get(user_id, [])))
        if len(pos) < 2 or not neg:
            continue
        for anchor in pos:
            same_cat_pos = [p for p in pos if p != anchor and int(world.category[p]) == int(world.category[anchor])]
            positive = rng.choice(same_cat_pos or [p for p in pos if p != anchor])
            # Hard negative: skipped item with largest oracle affinity for this user.
            candidates = rng.sample(neg, k=min(8, len(neg)))
            negative = max(candidates, key=lambda item: _click_logit(world, user_id, item))
            triplets.append((anchor, positive, negative))
            if len(triplets) >= max_triplets:
                return triplets
    return triplets


class CTRDataset(Dataset):
    def __init__(self, rows: Sequence[LogRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, item_id, clicked = self.rows[idx]
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "label": torch.tensor(clicked, dtype=torch.float32),
        }


class TripletDataset(Dataset):
    def __init__(self, triplets: Sequence[TripletRow]) -> None:
        self.triplets = list(triplets)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a, p, n = self.triplets[idx]
        return {
            "anchor": torch.tensor(a, dtype=torch.long),
            "positive": torch.tensor(p, dtype=torch.long),
            "negative": torch.tensor(n, dtype=torch.long),
        }


def make_batch(world: SyntheticWorld, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    item_id = batch["item_id"]
    return {
        "user_id": batch["user_id"],
        "item_id": item_id,
        "text": world.text_feat[item_id],
        "image": world.image_feat[item_id],
        "category": world.category[item_id],
        "price": world.price[item_id],
        "label": batch["label"],
    }


def make_triplet_batch(world: SyntheticWorld, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ("anchor", "positive", "negative"):
        item_id = batch[key]
        out[f"{key}_text"] = world.text_feat[item_id]
        out[f"{key}_image"] = world.image_feat[item_id]
    return out


def build_dataloaders(seed: int = 26, batch_size: int = 128) -> Tuple[SyntheticWorld, DataLoader, DataLoader, DataLoader, DataLoader]:
    world = build_world(seed=seed)
    train_rows, val_rows, test_rows = split_logs(build_logs(world, seed=seed))
    triplets = mine_triplets(train_rows, world, seed=seed)

    triplet_dl = DataLoader(TripletDataset(triplets), batch_size=batch_size, shuffle=True, num_workers=0)
    train_dl = DataLoader(CTRDataset(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(CTRDataset(val_rows), batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(CTRDataset(test_rows), batch_size=batch_size, shuffle=False, num_workers=0)
    return world, triplet_dl, train_dl, val_dl, test_dl


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu()
    labels = labels.detach().cpu().float()
    eps = 1e-7
    loss = -(labels * torch.log(probs + eps) + (1 - labels) * torch.log(1 - probs + eps)).mean().item()
    acc = ((probs >= 0.5).float() == labels).float().mean().item()

    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = 0.5
    else:
        scores = torch.cat([pos, neg])
        order = torch.argsort(scores)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
        auc = float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    topk = min(100, len(probs))
    top_idx = torch.topk(probs, k=topk).indices
    ctr_at_100 = labels[top_idx].mean().item()
    return {"auc": auc, "logloss": loss, "accuracy": acc, "ctr@100": ctr_at_100}
