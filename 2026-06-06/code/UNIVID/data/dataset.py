"""
Toy moderation dataset for UNIVID training.
Interface matches a real platform dataset:
    - Video frames (sampled uniformly)
    - Audio transcript token IDs
    - Policy category label(s)
    - Caption text (for teacher-forcing during SFT)
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import random


VIOLATION_CATEGORIES = [
    "violence", "adult_content", "spam", "misinformation",
    "hate_speech", "dangerous_activity", "copyright_violation",
    "underage_content", "self_harm", "commercial_fraud",
]


def _make_toy_frames(num_frames: int = 8) -> torch.Tensor:
    """Random RGB frames as placeholder."""
    return torch.randn(num_frames, 3, 224, 224)


def _make_toy_caption(category: str) -> str:
    templates = {
        "violence": "The video depicts physical altercation between two individuals with visible injury.",
        "spam": "Repetitive promotional content urging viewers to click external links.",
        "adult_content": "The video contains sexually suggestive material unsuitable for general audiences.",
        "misinformation": "The video promotes unverified health claims about a dietary supplement.",
    }
    return templates.get(category, f"The video contains content related to {category}.")


class ModerationDataset(Dataset):
    """
    Toy moderation dataset.
    Each sample contains:
        frames:         (T, 3, H, W)
        audio_embed:    (128,)   transcript embedding placeholder
        policy_ids:     (P,)     policy category indices
        caption_ids:    (L,)     tokenized caption (placeholder)
        labels:         (C,)     multi-label violation flags
        violation_flag: bool
    """

    def __init__(
        self,
        size: int = 1000,
        num_frames: int = 8,
        num_policies: int = 10,
        seq_len: int = 32,
        violation_rate: float = 0.3,
    ):
        self.size = size
        self.num_frames = num_frames
        self.num_policies = num_policies
        self.seq_len = seq_len
        self.violation_rate = violation_rate
        self._data = [self._generate_sample() for _ in range(size)]

    def _generate_sample(self) -> Dict[str, Any]:
        is_violation = random.random() < self.violation_rate
        violated_cats = []
        if is_violation:
            n_violations = random.randint(1, 2)
            violated_cats = random.sample(range(len(VIOLATION_CATEGORIES)), n_violations)

        labels = torch.zeros(len(VIOLATION_CATEGORIES))
        for c in violated_cats:
            labels[c] = 1.0

        return {
            "frames": _make_toy_frames(self.num_frames),
            "audio_embed": torch.randn(128),
            "policy_ids": torch.arange(self.num_policies),
            "caption_ids": torch.randint(0, 32000, (self.seq_len,)),
            "labels": labels,
            "violation_flag": torch.tensor(float(is_violation)),
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


def build_dataloader(size: int = 1000, batch_size: int = 8, num_workers: int = 0):
    dataset = ModerationDataset(size=size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
