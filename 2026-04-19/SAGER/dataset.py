from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class WorldConfig:
    num_users: int = 200
    num_items: int = 2000
    num_attrs: int = 24
    attrs_per_item: int = 3
    slate_size: int = 10
    rounds: int = 30
    seed: int = 42


class SyntheticWorld:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        rng = random.Random(cfg.seed)

        self.item_attrs = [sorted(rng.sample(range(cfg.num_attrs), cfg.attrs_per_item)) for _ in range(cfg.num_items)]

        g = torch.Generator().manual_seed(cfg.seed)
        self.user_pref = torch.randn(cfg.num_users, cfg.num_attrs, generator=g)

    def sample_candidates(self, rng: random.Random) -> list[int]:
        return rng.sample(range(self.cfg.num_items), self.cfg.slate_size)

    def true_utility(self, user_id: int, item_id: int) -> float:
        attrs = self.item_attrs[item_id]
        return float(self.user_pref[user_id, attrs].sum().item())

    def user_choice(self, user_id: int, slate: list[int]) -> int:
        best = max(slate, key=lambda it: self.true_utility(user_id, it))
        return best
