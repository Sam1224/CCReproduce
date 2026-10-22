from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class InteractionExample:
    history: list[int]
    target: int


class ToySequentialDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic sequential recommendation data.

    Items 0..head_cutoff-1 are frequent head items. Tail items are injected via
    cluster-specific transitions so a co-occurrence memory can recover them even
    when the neural generator under-ranks them.
    """

    def __init__(
        self,
        num_users: int = 512,
        num_items: int = 240,
        seq_len: int = 12,
        head_cutoff: int = 48,
        seed: int = 13,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.num_items = num_items
        self.seq_len = seq_len
        self.head_cutoff = head_cutoff
        self.examples: list[InteractionExample] = []

        cluster_count = 12
        tail_items = torch.arange(head_cutoff, num_items)
        clusters = tail_items.chunk(cluster_count)
        for user_id in range(num_users):
            cluster = clusters[user_id % cluster_count]
            head_context = torch.randint(0, head_cutoff, (seq_len - 3,), generator=generator).tolist()
            tail_context = cluster[torch.randint(0, len(cluster), (2,), generator=generator)].tolist()
            history = head_context + tail_context
            # 65% tail target, 35% head target, matching long-tail stress tests.
            if torch.rand((), generator=generator).item() < 0.65:
                target = int(cluster[torch.randint(0, len(cluster), (), generator=generator)].item())
            else:
                target = int(torch.randint(0, head_cutoff, (), generator=generator).item())
            self.examples.append(InteractionExample(history=history[-seq_len:], target=target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "history": torch.tensor(example.history, dtype=torch.long),
            "target": torch.tensor(example.target, dtype=torch.long),
        }


def split_dataset(dataset: ToySequentialDataset, ratio: float = 0.8) -> tuple[list[InteractionExample], list[InteractionExample]]:
    cut = int(len(dataset.examples) * ratio)
    return dataset.examples[:cut], dataset.examples[cut:]


def build_cooccurrence(
    examples: Iterable[InteractionExample],
    num_items: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return dense co-occurrence counts U[j, c].

    The paper uses sparse counts. A dense tensor keeps this reproduction compact;
    replace it with a sparse COO/CSR tensor for large catalogs.
    """

    matrix = torch.zeros((num_items, num_items), dtype=torch.float32, device=device)
    for example in examples:
        target = example.target
        for item in example.history:
            if item != target:
                matrix[item, target] += 1.0
                matrix[target, item] += 0.25
    return matrix


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "history": torch.stack([row["history"] for row in batch]),
        "target": torch.stack([row["target"] for row in batch]),
    }

