from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class SirfToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 512,
        vocab_size: int = 4096,
        num_policy_nodes: int = 64,
        seq_len_range: tuple[int, int] = (48, 160),
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.num_policy_nodes = num_policy_nodes
        self.seq_len_range = seq_len_range
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _segment(self, length: int, low: int, high: int) -> torch.Tensor:
        return torch.randint(low, high, (length,), dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec_len = self.rng.randint(8, 32)
        content_len = self.rng.randint(*self.seq_len_range)
        cot_len = self.rng.randint(8, 32)

        policy_node = self.rng.randrange(self.num_policy_nodes)
        risk_strength = self.rng.random()
        gray_band = 0.38 < risk_strength < 0.58
        verdict = 1 if gray_band else int(risk_strength >= 0.58) * 2

        spec = self._segment(spec_len, 16 + policy_node * 2, 16 + policy_node * 2 + 16).clamp_max(self.vocab_size - 1)
        content = self._segment(content_len, 256, self.vocab_size)
        cot = self._segment(cot_len, 128, 512)
        input_ids = torch.cat([torch.tensor([1]), spec, torch.tensor([2]), content, torch.tensor([3]), cot], dim=0)
        token_types = torch.cat([
            torch.zeros(1 + spec_len, dtype=torch.long),
            torch.ones(1 + content_len, dtype=torch.long),
            torch.full((1 + cot_len,), 2, dtype=torch.long),
        ])

        account_features = torch.tensor([
            risk_strength,
            self.rng.random(),
            self.rng.random(),
            float(verdict == 2),
            float(verdict == 1),
            self.rng.random(),
            self.rng.random(),
            self.rng.random(),
        ], dtype=torch.float)
        policy_path = torch.zeros(self.num_policy_nodes, dtype=torch.float)
        policy_path[policy_node] = 1.0
        for _ in range(self.rng.randint(1, 4)):
            policy_path[self.rng.randrange(self.num_policy_nodes)] = 1.0

        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "account_features": account_features,
            "policy_node": torch.tensor(policy_node, dtype=torch.long),
            "policy_path": policy_path,
            "verdict_label": torch.tensor(verdict, dtype=torch.long),
        }


def collate_sirf(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    token_types = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        token_types[row, :length] = item["token_types"]
        attention_mask[row, :length] = True

    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "attention_mask": attention_mask,
        "account_features": torch.stack([item["account_features"] for item in batch]),
        "policy_node": torch.stack([item["policy_node"] for item in batch]),
        "policy_path": torch.stack([item["policy_path"] for item in batch]),
        "verdict_label": torch.stack([item["verdict_label"] for item in batch]),
    }
