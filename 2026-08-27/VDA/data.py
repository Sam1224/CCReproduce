from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ContinualBatch:
    image_regions: torch.Tensor
    token_ids: torch.Tensor
    labels: torch.Tensor
    task_id: torch.Tensor
    visual_dependence: torch.Tensor


class VdaToyDataset(Dataset):
    def __init__(self, examples: List[Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        regions, tokens, label, task_id, vd = self.examples[idx]
        return {
            "image_regions": regions.float(),
            "token_ids": tokens.long(),
            "labels": torch.tensor(label, dtype=torch.long),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "visual_dependence": vd.float(),
        }


def build_task(task_id: int, seed: int, n: int, regions: int = 6, seq_len: int = 10, dim: int = 24, vocab: int = 60):
    gen = torch.Generator().manual_seed(seed + task_id * 17)
    prototypes = torch.randn(3, dim, generator=gen) + task_id * 0.15
    examples = []
    for _ in range(n):
        label = int(torch.randint(0, 3, (1,), generator=gen).item())
        image = prototypes[label].unsqueeze(0) + 0.4 * torch.randn(regions, dim, generator=gen)
        tokens = torch.randint(1, vocab, (seq_len,), generator=gen)
        tokens[:3] = label * 6 + torch.arange(1, 4)
        vd = torch.zeros(seq_len)
        vd[:3] = torch.tensor([0.95, 0.75, 0.60])
        vd[3:] = 0.10 + 0.10 * torch.rand(seq_len - 3, generator=gen)
        examples.append((image, tokens, label, task_id, vd))
    return examples


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0]}


def build_continual_loaders(tasks: int = 3, batch_size: int = 32):
    train_loaders, test_loaders = [], []
    for task_id in range(tasks):
        train = VdaToyDataset(build_task(task_id, 31, 360))
        test = VdaToyDataset(build_task(task_id, 91, 120))
        train_loaders.append(DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn))
        test_loaders.append(DataLoader(test, batch_size=batch_size, collate_fn=collate_fn))
    return train_loaders, test_loaders
