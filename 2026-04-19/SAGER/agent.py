from __future__ import annotations

import random

from dataset import SyntheticWorld
from skill import PolicySkill


class SagerAgent:
    def __init__(self, world: SyntheticWorld, seed: int = 0):
        self.world = world
        self.rng = random.Random(seed)
        self.full_skills = {u: PolicySkill.empty() for u in range(world.cfg.num_users)}

    def recommend(self, user_id: int, candidates: list[int]) -> list[int]:
        slim = self.full_skills[user_id].distill(max_attrs=5)
        scored = sorted(
            candidates,
            key=lambda it: slim.score_item(self.world.item_attrs[it]),
            reverse=True,
        )
        return scored

    def step(self, user_id: int) -> tuple[float, float]:
        candidates = self.world.sample_candidates(self.rng)
        ranked = self.recommend(user_id, candidates)

        chosen = self.world.user_choice(user_id, ranked)
        top1 = ranked[0]

        reward_hit1 = 1.0 if top1 == chosen else 0.0

        if reward_hit1 < 1.0:
            self.full_skills[user_id].update_reinforce_discover_weaken(
                chosen_attrs=self.world.item_attrs[chosen],
                rejected_attrs=self.world.item_attrs[top1],
                lr=0.1,
            )
        else:
            self.full_skills[user_id].update_reinforce_discover_weaken(
                chosen_attrs=self.world.item_attrs[chosen],
                rejected_attrs=[],
                lr=0.05,
            )

        return reward_hit1, float(len(self.full_skills[user_id].distill().weights))
