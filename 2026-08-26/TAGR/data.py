from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SyntheticLiveWorld:
    user_pref: torch.Tensor  # [U, D]
    user_price_sens: torch.Tensor  # [U]
    ad_emb: torch.Tensor  # [I, D]
    ad_category: torch.Tensor  # [I]
    ad_price: torch.Tensor  # [I]
    ad_quality: torch.Tensor  # [I]
    room_style: torch.Tensor  # [R, D]


def build_world(
    *,
    seed: int = 26,
    num_users: int = 600,
    num_ads: int = 500,
    num_rooms: int = 80,
    num_categories: int = 10,
    d: int = 32,
) -> SyntheticLiveWorld:
    g = torch.Generator().manual_seed(seed)
    return SyntheticLiveWorld(
        user_pref=torch.randn(num_users, d, generator=g) / math.sqrt(d),
        user_price_sens=torch.rand(num_users, generator=g),
        ad_emb=torch.randn(num_ads, d, generator=g) / math.sqrt(d),
        ad_category=torch.randint(0, num_categories, (num_ads,), generator=g),
        ad_price=torch.rand(num_ads, generator=g),
        ad_quality=torch.rand(num_ads, generator=g),
        room_style=torch.randn(num_rooms, d, generator=g) / math.sqrt(d),
    )


def _live_state(room_id: int, step: int, num_categories: int) -> Tuple[int, int, int, float]:
    # stage: warm-up -> selling -> flash-sale -> wrap-up; focus category drifts over time.
    stage = min(step // 8, 3)
    focus_cat = (room_id * 3 + step // 3) % num_categories
    promo = int((step % 11) > 6) + int(stage == 2)
    freshness = 1.0 - min(step, 31) / 31.0
    return stage, focus_cat, promo, freshness


def _sample_history(
    world: SyntheticLiveWorld,
    *,
    user_id: int,
    focus_cat: int,
    seq_len: int,
    g: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_ads = world.ad_emb.shape[0]
    cat_match = torch.where(world.ad_category == focus_cat)[0]
    hist: List[int] = []
    acts: List[int] = []
    for _ in range(seq_len):
        if len(cat_match) > 0 and torch.rand((), generator=g).item() < 0.65:
            item = int(cat_match[torch.randint(0, len(cat_match), (1,), generator=g)].item())
        else:
            item = int(torch.randint(0, num_ads, (1,), generator=g).item())
        score = (world.user_pref[user_id] * world.ad_emb[item]).sum() + 0.4 * world.ad_quality[item]
        act = int(torch.clamp(torch.floor((score + 0.6) * 2.0), 0, 3).item())  # view/click/cart/order
        hist.append(item)
        acts.append(act)
    return torch.tensor(hist, dtype=torch.long), torch.tensor(acts, dtype=torch.long)


def _oracle(
    world: SyntheticLiveWorld,
    *,
    user_id: int,
    room_id: int,
    cand: torch.Tensor,
    hist: torch.Tensor,
    act: torch.Tensor,
    stage: int,
    focus_cat: int,
    promo: int,
    freshness: float,
    g: torch.Generator,
) -> Tuple[int, torch.Tensor]:
    user = world.user_pref[user_id]
    room = world.room_style[room_id]
    hist_emb = world.ad_emb[hist]
    act_w = (act.float() + 1.0) / 4.0
    intent = (hist_emb * act_w[:, None]).sum(0) / act_w.sum().clamp_min(1.0)

    item = world.ad_emb[cand]
    cat = world.ad_category[cand]
    price = world.ad_price[cand]
    quality = world.ad_quality[cand]
    cat_bonus = (cat == focus_cat).float()

    live_shift = 0.35 * room + 0.25 * intent + 0.15 * stage * user
    relevance = item @ (user + live_shift)
    conversion = quality - world.user_price_sens[user_id] * price + 0.35 * promo * cat_bonus
    recency = freshness * cat_bonus + 0.15 * (act.float().mean() / 3.0)
    reward_proxy = torch.sigmoid(relevance + conversion + recency)
    noisy = reward_proxy + 0.03 * torch.randn(reward_proxy.shape, generator=g)
    return int(torch.argmax(noisy).item()), reward_proxy


class TagrDataset(Dataset):
    def __init__(self, world: SyntheticLiveWorld, examples: List[Dict[str, torch.Tensor]]) -> None:
        self.world = world
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def build_dataloaders(
    *,
    seed: int = 26,
    batch_size: int = 96,
    seq_len: int = 12,
    candidate_size: int = 32,
    train_examples: int = 2400,
    val_examples: int = 400,
    test_examples: int = 400,
) -> Tuple[SyntheticLiveWorld, DataLoader, DataLoader, DataLoader]:
    world = build_world(seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    num_users, num_ads, num_rooms = world.user_pref.shape[0], world.ad_emb.shape[0], world.room_style.shape[0]
    num_categories = int(world.ad_category.max().item()) + 1

    def make_examples(n: int) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for i in range(n):
            user_id = int(torch.randint(0, num_users, (1,), generator=g).item())
            room_id = int(torch.randint(0, num_rooms, (1,), generator=g).item())
            step = int(torch.randint(0, 32, (1,), generator=g).item())
            stage, focus_cat, promo, freshness = _live_state(room_id, step, num_categories)
            hist, act = _sample_history(world, user_id=user_id, focus_cat=focus_cat, seq_len=seq_len, g=g)

            focus_pool = torch.where(world.ad_category == focus_cat)[0]
            random_part = torch.randint(0, num_ads, (candidate_size,), generator=g)
            if len(focus_pool) > 0:
                focus_part = focus_pool[torch.randint(0, len(focus_pool), (candidate_size,), generator=g)]
                cand = torch.cat([random_part[: candidate_size // 2], focus_part[: candidate_size - candidate_size // 2]])
            else:
                cand = random_part
            label, reward = _oracle(
                world,
                user_id=user_id,
                room_id=room_id,
                cand=cand,
                hist=hist,
                act=act,
                stage=stage,
                focus_cat=focus_cat,
                promo=promo,
                freshness=freshness,
                g=g,
            )
            out.append(
                {
                    "user_id": torch.tensor(user_id, dtype=torch.long),
                    "room_id": torch.tensor(room_id, dtype=torch.long),
                    "hist_item_ids": hist,
                    "hist_actions": act,
                    "cand_item_ids": cand.to(torch.long),
                    "stage": torch.tensor(stage, dtype=torch.long),
                    "focus_cat": torch.tensor(focus_cat, dtype=torch.long),
                    "promo": torch.tensor(promo, dtype=torch.long),
                    "freshness": torch.tensor(freshness, dtype=torch.float),
                    "label": torch.tensor(label, dtype=torch.long),
                    "reward_proxy": reward.to(torch.float),
                }
            )
        return out

    train = TagrDataset(world, make_examples(train_examples))
    val = TagrDataset(world, make_examples(val_examples))
    test = TagrDataset(world, make_examples(test_examples))
    return (
        world,
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn),
    )


def hr_ndcg(scores: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    topk = torch.topk(scores, k=max(ks), dim=-1).indices
    metrics: Dict[str, float] = {}
    bsz = scores.shape[0]
    for k in ks:
        hit = (topk[:, :k] == labels[:, None]).any(dim=-1).float().mean().item()
        pos = torch.where(topk[:, :k] == labels[:, None])
        ndcg = torch.zeros(bsz)
        if pos[0].numel() > 0:
            ndcg[pos[0]] = 1.0 / torch.log2(pos[1].float() + 2.0)
        metrics[f"hr@{k}"] = hit
        metrics[f"ndcg@{k}"] = ndcg.mean().item()
    return metrics
