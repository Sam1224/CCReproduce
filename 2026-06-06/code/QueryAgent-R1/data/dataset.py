"""
Toy e-commerce query recommendation dataset for QueryAgent-R1.
Interface mirrors a real industrial dataset:
    - interaction_history: user's past interacted product embeddings
    - current_context:     current session product/query embedding
    - query_ids:           ground-truth query token IDs (for SFT warmup)
    - ctr_label:           click label for the query (for CTR estimator training)
"""

import torch
from torch.utils.data import Dataset
from typing import Dict


class EComQueryDataset(Dataset):
    """
    Toy e-commerce dataset for QueryAgent-R1.

    Each sample represents a user session with:
        - Interaction history (M past product embeddings)
        - Current context embedding
        - A recommended query (token IDs)
        - A CTR label (0/1)
    """

    def __init__(
        self,
        size: int = 2000,
        memory_len: int = 20,
        embed_dim: int = 64,
        query_vocab: int = 10_000,
        query_len: int = 8,
        ctr_rate: float = 0.4,
    ):
        self.size = size
        self.memory_len = memory_len
        self.embed_dim = embed_dim
        self.query_vocab = query_vocab
        self.query_len = query_len
        self.ctr_rate = ctr_rate
        import random
        self._data = [self._generate(random) for _ in range(size)]

    def _generate(self, rng) -> Dict[str, torch.Tensor]:
        return {
            "interaction_history": torch.randn(self.memory_len, self.embed_dim),
            "current_context": torch.randn(self.embed_dim),
            "query_ids": torch.randint(1, self.query_vocab, (self.query_len,)),
            "ctr_label": torch.tensor(float(rng.random() < self.ctr_rate)),
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


def build_dataloader(size: int = 2000, batch_size: int = 16, num_workers: int = 0):
    dataset = EComQueryDataset(size=size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
