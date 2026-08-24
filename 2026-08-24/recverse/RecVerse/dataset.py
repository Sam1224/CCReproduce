import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


ACTIONS = ["scroll_down", "scroll_up", "click", "enter_detail", "go_back", "add_to_cart", "purchase", "terminate", "memorize"]
CATEGORIES = ["apparel", "food", "electronics", "family", "beauty"]


class ShoppingTrajectoryDataset(Dataset):
    def __init__(self, trajectories: int = 160, steps: int = 8, seed: int = 23):
        random.seed(seed)
        self.rows: List[Dict[str, torch.Tensor]] = []
        for _ in range(trajectories):
            preference = random.randrange(len(CATEGORIES))
            screen_features = torch.randn(steps, 32)
            category_signal = torch.zeros(steps, len(CATEGORIES))
            actions = []
            for step in range(steps):
                shown_category = preference if random.random() < 0.55 else random.randrange(len(CATEGORIES))
                category_signal[step, shown_category] = 1.0
                if shown_category == preference and step > 2:
                    action = random.choice([2, 3, 5, 6])
                elif step == steps - 1:
                    action = 7
                else:
                    action = random.choice([0, 0, 2, 8])
                actions.append(action)
            self.rows.append({
                "screen": screen_features,
                "category": category_signal,
                "preference": torch.tensor(preference, dtype=torch.long),
                "actions": torch.tensor(actions, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]
