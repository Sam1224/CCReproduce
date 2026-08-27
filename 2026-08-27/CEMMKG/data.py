from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


VOCAB = [
    "shirt", "shoe", "bag", "cosmetic", "phone", "discount", "quality", "style",
    "red", "blue", "black", "summer", "luxury", "review", "delivery", "creator",
    "caption", "image", "policy", "duplicate", "brand", "material", "size", "trend",
]


@dataclass
class VisualElement:
    doc_id: int
    image_id: int
    feature: torch.Tensor
    surrounding_text: List[int]
    semantic_text: List[int]
    global_text: List[int]
    label: int


class CemmkgDataset(Dataset):
    def __init__(self, examples: List[VisualElement]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "feature": ex.feature.float(),
            "surrounding": torch.tensor(ex.surrounding_text, dtype=torch.long),
            "semantic": torch.tensor(ex.semantic_text, dtype=torch.long),
            "global_text": torch.tensor(ex.global_text, dtype=torch.long),
            "label": torch.tensor(ex.label, dtype=torch.long),
        }


def _tokens(rng: random.Random, center: int, length: int, vocab_size: int) -> List[int]:
    out = []
    for _ in range(length):
        if rng.random() < 0.55:
            out.append((center + rng.randint(-2, 2)) % vocab_size)
        else:
            out.append(rng.randrange(vocab_size))
    return out


def build_examples(seed: int, n: int, image_dim: int = 32, num_classes: int = 6) -> List[VisualElement]:
    rng = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(num_classes, image_dim, generator=torch_gen)
    examples: List[VisualElement] = []
    for idx in range(n):
        label = rng.randrange(num_classes)
        feature = prototypes[label] + 0.35 * torch.randn(image_dim, generator=torch_gen)
        center = (label * 4) % len(VOCAB)
        examples.append(
            VisualElement(
                doc_id=idx // 4,
                image_id=idx,
                feature=feature,
                surrounding_text=_tokens(rng, center, 8, len(VOCAB)),
                semantic_text=_tokens(rng, center + 1, 12, len(VOCAB)),
                global_text=_tokens(rng, center + 2, 16, len(VOCAB)),
                label=label,
            )
        )
    return examples


def build_dataloaders(batch_size: int = 64):
    train = CemmkgDataset(build_examples(13, 1500))
    val = CemmkgDataset(build_examples(14, 300))
    test = CemmkgDataset(build_examples(15, 300))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
