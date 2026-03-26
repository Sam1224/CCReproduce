from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class CuaStep:
    frames: torch.Tensor  # (T,C,H,W)
    cursor_xy: torch.Tensor  # (2,)
    action_id: torch.Tensor  # ()
    reasoning_ids: torch.Tensor  # (Lr,)


class VideoCuaToyDataset(Dataset):
    """Toy version of CUA-Suite dataset.

    The real dataset contains 30fps continuous videos + cursor traces + dense reasoning annotations.
    Here we only ensure the *interfaces* exist and tensors have legal shapes.
    """

    def __init__(
        self,
        num_samples: int = 256,
        num_frames: int = 4,
        image_size: int = 64,
        num_actions: int = 12,
        vocab_size: int = 2000,
        reasoning_len: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_actions = num_actions
        self.vocab_size = vocab_size
        self.reasoning_len = reasoning_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frames = torch.randn(self.num_frames, 3, self.image_size, self.image_size)
        cursor_xy = torch.rand(2)
        action_id = torch.randint(0, self.num_actions, (1,), dtype=torch.long).squeeze(0)
        reasoning_ids = torch.randint(0, self.vocab_size, (self.reasoning_len,), dtype=torch.long)
        return {"frames": frames, "cursor_xy": cursor_xy, "action_id": action_id, "reasoning_ids": reasoning_ids}


class ScreenshotCuaToyDataset(VideoCuaToyDataset):
    """Adapter: convert video steps to single-frame steps."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        # Use the first frame as screenshot.
        sample["frames"] = sample["frames"][:1]
        return sample


def collate_cua(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B,T,C,H,W)
    cursor_xy = torch.stack([b["cursor_xy"] for b in batch], dim=0)  # (B,2)
    action_id = torch.stack([b["action_id"] for b in batch], dim=0)  # (B,)
    reasoning_ids = torch.stack([b["reasoning_ids"] for b in batch], dim=0)  # (B,Lr)
    return {"frames": frames, "cursor_xy": cursor_xy, "action_id": action_id, "reasoning_ids": reasoning_ids}
