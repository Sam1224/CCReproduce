from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicySkill:
    # Attribute -> weight
    weights: dict[int, float]

    @staticmethod
    def empty() -> "PolicySkill":
        return PolicySkill(weights={})

    def distill(self, max_attrs: int = 5) -> "PolicySkill":
        if not self.weights:
            return PolicySkill.empty()
        top = sorted(self.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_attrs]
        return PolicySkill(weights=dict(top))

    def score_item(self, item_attrs: list[int]) -> float:
        return sum(self.weights.get(a, 0.0) for a in item_attrs)

    def update_reinforce_discover_weaken(
        self,
        chosen_attrs: list[int],
        rejected_attrs: list[int],
        lr: float = 0.1,
    ) -> None:
        chosen_set = set(chosen_attrs)
        rejected_set = set(rejected_attrs)

        for a in chosen_set:
            self.weights[a] = self.weights.get(a, 0.0) + lr

        for a in rejected_set - chosen_set:
            self.weights[a] = self.weights.get(a, 0.0) - lr

        # Discover: amplify attributes that are consistently present in choices.
        for a in chosen_set & rejected_set:
            self.weights[a] = self.weights.get(a, 0.0) + 0.5 * lr
