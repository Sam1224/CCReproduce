from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


VIOLATION_CATEGORIES = [
    "benign",
    "false_advertising",
    "medical_claim",
    "counterfeit",
    "prohibited_item",
    "privacy_leak",
    "traffic_diversion",
]

EVASIVE_PATTERNS = ["clean", "split_word", "euphemism", "homophone", "cropped_visual", "masked_claim"]

VOCAB = {
    "benign": 1,
    "official": 2,
    "discount": 3,
    "review": 4,
    "guaranteed": 5,
    "cure": 6,
    "replica": 7,
    "contact": 8,
    "private": 9,
    "weapon": 10,
    "wechat": 11,
    "best": 12,
    "safe": 13,
    "claim": 14,
    "image": 15,
    "cropped": 16,
    "mask": 17,
    "split": 18,
    "hint": 19,
    "normal": 20,
}


@dataclass(frozen=True)
class EvasiveSample:
    text_ids: torch.Tensor
    image_features: torch.Tensor
    rule_id: torch.Tensor
    label: torch.Tensor
    is_evasive: torch.Tensor


def _tokens_for_label(label: int, pattern: str) -> List[int]:
    base_tokens = [VOCAB["official"], VOCAB["discount"], VOCAB["review"]]
    category_tokens = {
        0: [VOCAB["benign"], VOCAB["normal"]],
        1: [VOCAB["guaranteed"], VOCAB["best"], VOCAB["claim"]],
        2: [VOCAB["cure"], VOCAB["safe"], VOCAB["claim"]],
        3: [VOCAB["replica"], VOCAB["official"]],
        4: [VOCAB["weapon"], VOCAB["contact"]],
        5: [VOCAB["private"], VOCAB["contact"]],
        6: [VOCAB["wechat"], VOCAB["contact"]],
    }[label]
    pattern_tokens = {
        "clean": [],
        "split_word": [VOCAB["split"], VOCAB["hint"]],
        "euphemism": [VOCAB["hint"], VOCAB["safe"]],
        "homophone": [VOCAB["hint"], VOCAB["claim"]],
        "cropped_visual": [VOCAB["image"], VOCAB["cropped"]],
        "masked_claim": [VOCAB["mask"], VOCAB["claim"]],
    }[pattern]
    return base_tokens + category_tokens + pattern_tokens


class SyntheticEvasiveContentDataset(Dataset):
    def __init__(self, size: int = 512, seq_len: int = 24, image_dim: int = 32, seed: int = 7):
        self.size = size
        self.seq_len = seq_len
        self.image_dim = image_dim
        self.rng = random.Random(seed)
        self.samples = [self._make_sample() for _ in range(size)]

    def _make_sample(self) -> EvasiveSample:
        label = self.rng.randint(0, len(VIOLATION_CATEGORIES) - 1)
        pattern = "clean" if label == 0 else self.rng.choice(EVASIVE_PATTERNS[1:])
        tokens = _tokens_for_label(label, pattern)
        while len(tokens) < self.seq_len:
            tokens.append(self.rng.randint(1, max(VOCAB.values())))
        self.rng.shuffle(tokens)
        text_ids = torch.tensor(tokens[: self.seq_len], dtype=torch.long)

        image_features = torch.randn(self.image_dim) * 0.15
        if label > 0:
            image_features[label % self.image_dim] += 2.0
        if pattern in {"cropped_visual", "masked_claim"}:
            image_features[-1] += 1.5

        return EvasiveSample(
            text_ids=text_ids,
            image_features=image_features.float(),
            rule_id=torch.tensor(label, dtype=torch.long),
            label=torch.tensor(label, dtype=torch.long),
            is_evasive=torch.tensor(1 if pattern != "clean" else 0, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "text_ids": sample.text_ids,
            "image_features": sample.image_features,
            "rule_id": sample.rule_id,
            "label": sample.label,
            "is_evasive": sample.is_evasive,
        }
