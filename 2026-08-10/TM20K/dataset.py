import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class ECommerceSequenceConfig:
    num_items: int = 2048
    num_categories: int = 64
    max_seq_len: int = 512
    latent_dim: int = 16
    seed: int = 20260810


class SyntheticECommerceDataset(Dataset):
    def __init__(self, size: int = 4096, config: ECommerceSequenceConfig | None = None):
        self.size = size
        self.config = config or ECommerceSequenceConfig()
        generator = torch.Generator().manual_seed(self.config.seed)
        self.item_category = torch.randint(
            0, self.config.num_categories, (self.config.num_items,), generator=generator
        )
        self.item_quality = torch.randn(self.config.num_items, generator=generator) * 0.35
        self.category_affinity = torch.randn(
            self.config.num_categories, self.config.latent_dim, generator=generator
        )
        self.user_interests = torch.randn(size, self.config.latent_dim, generator=generator)
        self.samples = [self._make_sample(index, generator) for index in range(size)]

    def _make_sample(self, index: int, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        cfg = self.config
        user_interest = self.user_interests[index]
        target_item = torch.randint(1, cfg.num_items, (1,), generator=generator).item()
        target_category = self.item_category[target_item]
        logits = self.category_affinity @ user_interest
        probs = torch.softmax(logits, dim=0)
        categories = torch.multinomial(probs, cfg.max_seq_len, replacement=True, generator=generator)
        noise_items = torch.randint(1, cfg.num_items, (cfg.max_seq_len,), generator=generator)
        category_items = (categories * math.ceil(cfg.num_items / cfg.num_categories)) % cfg.num_items
        switch = torch.rand(cfg.max_seq_len, generator=generator) < 0.72
        sequence = torch.where(switch, category_items, noise_items).clamp_min(1)
        recency = torch.linspace(0.2, 1.0, cfg.max_seq_len)
        match = (self.item_category[sequence] == target_category).float()
        score = (match * recency).mean() * 5.0 + self.item_quality[target_item]
        label = (torch.sigmoid(score - 1.15) > torch.rand((), generator=generator)).float()
        return {
            "sequence": sequence.long(),
            "target": torch.tensor(target_item, dtype=torch.long),
            "label": label.float(),
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]
