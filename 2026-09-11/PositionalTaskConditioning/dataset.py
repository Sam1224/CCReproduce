from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


TASKS = ["duplicate_variant", "unit_mismatch", "theme_defect", "attribute_overstuffing"]


class ProductFamilyToyDataset(Dataset):
    def __init__(self, num_samples: int = 512, vocab_size: int = 2048, max_items: int = 32, seed: int = 0) -> None:
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_items = max_items
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_id = self.rng.randrange(len(TASKS))
        length = self.rng.randint(8, self.max_items)
        base = self.rng.randint(32, self.vocab_size - 128)
        ids = torch.randint(base, min(base + 96, self.vocab_size), (length,), dtype=torch.long)

        defect_strength = self.rng.random()
        if task_id == 0 and defect_strength > 0.55:
            ids[-2:] = ids[:2]
        elif task_id == 1 and defect_strength > 0.55:
            ids[::5] = 17
        elif task_id == 2 and defect_strength > 0.55:
            ids[length // 2 :] = torch.randint(1500, self.vocab_size, (length - length // 2,), dtype=torch.long)
        elif task_id == 3 and defect_strength > 0.55:
            extra = torch.tensor([19, 23, 29, 31], dtype=torch.long)
            ids[: min(length, 4)] = extra[: min(length, 4)]

        label = 2 if defect_strength > 0.68 else 1 if defect_strength > 0.45 else 0
        teacher_probs = torch.full((3,), 0.05)
        teacher_probs[label] = 0.90
        return {
            "item_ids": ids,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "teacher_probs": teacher_probs,
        }


def collate_ptc(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item["item_ids"].numel() for item in batch)
    item_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    item_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["item_ids"].numel()
        item_ids[row, :length] = item["item_ids"]
        item_mask[row, :length] = True
    return {
        "item_ids": item_ids,
        "item_mask": item_mask,
        "task_id": torch.stack([item["task_id"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "teacher_probs": torch.stack([item["teacher_probs"] for item in batch]),
    }
