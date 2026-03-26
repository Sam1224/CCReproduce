from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

"""
Paper: CUA-Suite: Massive Human-Annotated Video Demonstrations for Computer-Use Agents
Summary: 30fps continuous videos + cursor traces + dense multi-layer reasoning annotations.
Core: Provide a unified dataset format that can be converted into screenshot-based or video-based agent training.

Note:
This is a minimal dataset abstraction capturing the essential modalities.
"""


@dataclass
class CuaStep:
    frames: torch.Tensor  # (T, C, H, W)
    cursor_xy: torch.Tensor  # (T, 2)
    action_id: torch.Tensor  # scalar
    reasoning_text: Optional[str] = None


class VideoCuaDataset(Dataset):
    def __init__(self, steps: List[CuaStep]):
        self.steps = steps

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> CuaStep:
        return self.steps[idx]


def collate_video_cua(batch: List[CuaStep]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = torch.stack([b.frames for b in batch], dim=0)
    cursor = torch.stack([b.cursor_xy for b in batch], dim=0)
    action = torch.stack([b.action_id for b in batch], dim=0).long().view(-1)
    return frames, cursor, action


if __name__ == "__main__":
    torch.manual_seed(0)

    fake_steps: List[CuaStep] = []
    for _ in range(4):
        fake_steps.append(
            CuaStep(
                frames=torch.randn(12, 3, 128, 128),
                cursor_xy=torch.rand(12, 2),
                action_id=torch.randint(0, 16, (1,)),
                reasoning_text="click the menu then open settings",
            )
        )

    ds = VideoCuaDataset(fake_steps)
    frames, cursor, action = collate_video_cua([ds[0], ds[1]])

    print("frames:", tuple(frames.shape))
    print("cursor:", tuple(cursor.shape))
    print("action:", action.tolist())
    print("CUA-Suite dataset reproduction structure complete.")
