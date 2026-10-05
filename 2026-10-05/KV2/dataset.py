import random
from typing import Dict

import torch
from torch.utils.data import Dataset


class NeedleDataset(Dataset):
    def __init__(self, size: int = 2048, seq_len: int = 96, vocab_size: int = 128, seed: int = 11):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        rng = random.Random(seed)
        self.samples = []
        for _ in range(size):
            key = rng.randrange(4, vocab_size)
            token_pool = [token for token in range(4, vocab_size) if token != key]
            tokens = [rng.choice(token_pool) for _ in range(seq_len)]
            value = key
            needle_pos = rng.randrange(4, seq_len - 4)
            tokens[needle_pos] = key
            query = key
            self.samples.append((tokens, query, value, needle_pos))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        tokens, query, value, needle_pos = self.samples[index]
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "query": torch.tensor(query, dtype=torch.long),
            "label": torch.tensor(value, dtype=torch.long),
            "needle_pos": torch.tensor(needle_pos, dtype=torch.long),
        }
