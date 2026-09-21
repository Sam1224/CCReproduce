from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


ROLES = ["project_manager", "finance_analyst", "trust_ops", "merchant_growth"]
FIELDS = {
    "project_manager": ["deadline", "owner", "blocker"],
    "finance_analyst": ["budget", "roi", "risk"],
    "trust_ops": ["policy", "evidence", "severity"],
    "merchant_growth": ["segment", "campaign", "conversion"],
}


@dataclass
class Feedback:
    role: str
    missing_field: str
    score_delta: float


class PersonaIEDataset(Dataset):
    def __init__(self, size: int = 240, seed: int = 23):
        rng = random.Random(seed)
        rows = []
        for _ in range(size):
            role = rng.choice(ROLES)
            target = rng.choice(FIELDS[role])
            rows.append((role, target, rng.random()))
        self.rows = rows
        self.role_to_id = {role: i for i, role in enumerate(ROLES)}
        all_fields = sorted({field for values in FIELDS.values() for field in values})
        self.field_to_id = {field: i for i, field in enumerate(all_fields)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        role, field, noise = self.rows[idx]
        return {
            "role_id": torch.tensor(self.role_to_id[role]),
            "target_id": torch.tensor(self.field_to_id[field]),
            "role": role,
            "field": field,
            "noise": torch.tensor(noise, dtype=torch.float32),
        }
