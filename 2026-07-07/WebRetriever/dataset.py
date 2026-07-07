from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class WebTaskConfig:
    samples: int = 256
    pages_per_site: int = 8
    feature_dim: int = 32
    action_steps: int = 5
    action_vocab: int = 6
    seed: int = 7


class SyntheticWebRetrieverDataset(Dataset):
    """Synthetic page-graph tasks following the WebRetriever benchmark interface."""

    def __init__(self, config: WebTaskConfig):
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        self.query = torch.randn(config.samples, config.feature_dim, generator=generator)
        self.pages = torch.randn(config.samples, config.pages_per_site, config.feature_dim, generator=generator)
        self.evidence_bias = torch.randn(config.samples, config.feature_dim, generator=generator) * 0.15
        self.target_page = torch.randint(0, config.pages_per_site, (config.samples,), generator=generator)
        self.actions = torch.randint(0, config.action_vocab, (config.samples, config.action_steps), generator=generator)

        for sample_index in range(config.samples):
            page_index = int(self.target_page[sample_index])
            self.pages[sample_index, page_index] = self.query[sample_index] + self.evidence_bias[sample_index]
            self.actions[sample_index, -1] = 0

        page_alignment = torch.cosine_similarity(
            self.query.unsqueeze(1), self.pages, dim=-1
        ).max(dim=1).values
        action_quality = (self.actions[:, -1] == 0).float()
        self.completed = ((page_alignment > 0.55) & (action_quality > 0)).float()

    def __len__(self) -> int:
        return self.config.samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "query": self.query[index],
            "pages": self.pages[index],
            "target_page": self.target_page[index],
            "actions": self.actions[index],
            "completed": self.completed[index],
        }
