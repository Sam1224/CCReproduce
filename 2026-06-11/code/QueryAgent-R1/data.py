"""
QueryAgent-R1 Data Module
Paper: arxiv 2606.05671

Data format:
  - User profile: {user_id, long-term history (behavior sequences)}
  - Query record: {user_id, search_intent, target_query, clicked_products}
  - Product catalog: {product_id, title, category, price, embedding}
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class UserSample:
    user_id: str
    search_intent: str          # User's expressed search need (text)
    target_query: str           # Ground truth "good" query (for SFT)
    clicked_products: List[str] # Product IDs user clicked (for reward)
    history_categories: List[int]  # Long-term browsing category indices
    history_behaviors: List[int]   # Behavior types (0=click, 1=purchase, etc.)
    history_weights: List[float]   # Engagement weights


class EcommerceQueryDataset(Dataset):
    """
    E-commerce query recommendation dataset.
    
    Industrial dataset (from paper): ~10M user samples from Alibaba International platform
    Public dataset: adapted from public e-commerce interaction logs
    
    Toy version: synthetic data for interface validation
    """

    BEHAVIOR_MAP = {
        "click": 0, "purchase": 1, "cart": 2, "wishlist": 3, "view": 4
    }

    INTENT_TEMPLATES = [
        "I'm looking for {adj} {category} items under ${price}",
        "Find me {adj} {category} for {occasion}",
        "Show me best selling {category} from top brands",
        "I need {category} similar to what I bought before",
        "Looking for {adj} {category} with good reviews",
    ]

    CATEGORIES = [
        "electronics", "clothing", "shoes", "home decor", "books",
        "sports equipment", "beauty products", "kitchen appliances",
        "toys", "jewelry", "bags", "watches", "pet supplies", "food",
        "office supplies", "garden tools", "musical instruments",
        "automotive parts", "baby products", "health supplements",
    ]

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        toy_size: int = 5000,
        max_history_len: int = 50,
    ):
        self.max_history_len = max_history_len

        if data_path is None:
            self.samples = self._generate_toy_data(toy_size)
        else:
            with open(f"{data_path}/{split}.json") as f:
                self.samples = json.load(f)

    def _generate_toy_data(self, n: int) -> List[UserSample]:
        samples = []
        for i in range(n):
            # Random user history (5-50 items)
            hist_len = random.randint(5, self.max_history_len)
            hist_cats = [random.randint(0, len(self.CATEGORIES)-1) for _ in range(hist_len)]
            hist_behav = [random.randint(0, 4) for _ in range(hist_len)]
            hist_weights = [0.5 + random.random() * 0.5 for _ in range(hist_len)]
            # Normalize
            total_w = sum(hist_weights)
            hist_weights = [w / total_w for w in hist_weights]

            # Dominant category
            from collections import Counter
            dominant_cat_idx = Counter(hist_cats).most_common(1)[0][0]
            dominant_cat = self.CATEGORIES[dominant_cat_idx]

            adj = random.choice(["cheap", "premium", "vintage", "modern", "eco-friendly"])
            occasion = random.choice(["everyday use", "gifting", "travel", "home"])
            price = random.choice([20, 50, 100, 200, 500])

            intent_tmpl = random.choice(self.INTENT_TEMPLATES)
            intent = intent_tmpl.format(
                adj=adj, category=dominant_cat, price=price, occasion=occasion
            )
            target_query = f"{adj} {dominant_cat} best quality"

            samples.append(UserSample(
                user_id=f"user_{i:07d}",
                search_intent=intent,
                target_query=target_query,
                clicked_products=[f"pid_{random.randint(0, 9999):06d}" for _ in range(5)],
                history_categories=hist_cats,
                history_behaviors=hist_behav,
                history_weights=hist_weights,
            ))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        L = min(len(s.history_categories), self.max_history_len)

        cats = torch.zeros(self.max_history_len, dtype=torch.long)
        cats[:L] = torch.tensor(s.history_categories[:L])

        behaviors = torch.zeros(self.max_history_len, dtype=torch.long)
        behaviors[:L] = torch.tensor(s.history_behaviors[:L])

        weights = torch.zeros(self.max_history_len)
        weights[:L] = torch.tensor(s.history_weights[:L])

        mask = torch.zeros(self.max_history_len, dtype=torch.bool)
        mask[:L] = True

        return {
            "user_id": s.user_id,
            "search_intent": s.search_intent,
            "target_query": s.target_query,
            "clicked_products": s.clicked_products,
            "history_categories": cats,
            "history_behaviors": behaviors,
            "history_weights": weights,
            "history_mask": mask,
        }


def get_dataloader(
    data_path: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    dataset = EcommerceQueryDataset(data_path, split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )
