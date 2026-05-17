"""
Toy Livestream Dataset for Dual-Path Moderation Training
Interface-aligned with the production dataset from arXiv 2512.03553.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import random

from model.dual_path import VIOLATION_CLASSES, NUM_CLASSES


class ToyLivestreamDataset(Dataset):
    """
    Toy dataset simulating multimodal livestream content.
    Each sample: (video_frame, audio_mel, text_tokens, label, is_violation)

    Production dataset characteristics (paper):
      - Large-scale user-generated livestream clips
      - ~100K+ violation examples across violation types
      - MLLM-annotated soft labels for distillation
    """

    def __init__(
        self,
        num_samples: int = 200,
        seq_len: int = 64,
        img_size: int = 64,
        audio_frames: int = 500,
        vocab_size: int = 32000,
        violation_rate: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        random.seed(seed)
        torch.manual_seed(seed)
        self.samples = self._generate(
            num_samples, seq_len, img_size, audio_frames, vocab_size, violation_rate
        )

    def _generate(self, N, seq_len, img_size, audio_frames, vocab_size, vrate):
        samples = []
        for i in range(N):
            is_violation = random.random() < vrate
            if is_violation:
                label = random.randint(1, NUM_CLASSES - 1)  # non-compliant
            else:
                label = 0  # compliant

            # Simulate MLLM soft labels (one-hot smoothed for toy)
            hard = torch.zeros(NUM_CLASSES)
            hard[label] = 1.0
            soft_label = hard * 0.8 + torch.full((NUM_CLASSES,), 0.2 / NUM_CLASSES)

            sample = {
                "video_frames": torch.randn(3, img_size, img_size),
                "text_ids": torch.randint(0, vocab_size, (seq_len,)),
                "audio_mel": torch.randn(80, audio_frames),
                "hard_label": torch.tensor(label),
                "soft_label": soft_label,
                "is_violation": torch.tensor(float(is_violation)),
                "violation_type": VIOLATION_CLASSES[label],
            }
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key == "violation_type":
            result[key] = [s[key] for s in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([s[key] for s in batch])
    return result
