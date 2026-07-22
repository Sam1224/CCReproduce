import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class ToyConfig:
    image_size: int = 64
    num_messages: int = 6
    samples_per_message: int = 80
    hidden_strength: float = 0.18
    noise_std: float = 0.18
    seed: int = 7


def _meshgrid(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def make_template(message_id: int, size: int = 64) -> torch.Tensor:
    xx, yy = _meshgrid(size)
    pattern = torch.zeros(size, size)
    kind = message_id % 6
    if kind == 0:
        pattern = ((xx.abs() < 0.12) | (yy.abs() < 0.12)).float()
    elif kind == 1:
        pattern = (((xx + yy).abs() < 0.10) | ((xx - yy).abs() < 0.10)).float()
    elif kind == 2:
        radius = torch.sqrt(xx.square() + yy.square())
        pattern = ((radius > 0.42) & (radius < 0.58)).float()
    elif kind == 3:
        pattern = ((yy > -0.55) & (yy < -0.38) & (xx.abs() < 0.55)).float()
        pattern += ((xx > -0.55) & (xx < -0.38) & (yy.abs() < 0.55)).float()
        pattern += ((xx - yy).abs() < 0.08).float()
        pattern = pattern.clamp(0, 1)
    elif kind == 4:
        pattern = ((torch.sin(12 * math.pi * xx) > 0.86) | (torch.cos(12 * math.pi * yy) > 0.86)).float()
    else:
        pattern = (((xx + 0.35).square() + (yy + 0.30).square() < 0.10) | ((xx - 0.30).square() + (yy - 0.25).square() < 0.12)).float()
    return pattern.unsqueeze(0).repeat(3, 1, 1)


def make_background(size: int, generator: torch.Generator) -> torch.Tensor:
    base = torch.rand(3, size, size, generator=generator)
    for _ in range(3):
        base = torch.nn.functional.avg_pool2d(base.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
    color = torch.rand(3, 1, 1, generator=generator) * 0.7 + 0.2
    return (0.65 * base + 0.35 * color).clamp(0, 1)


class HiddenMessageDataset(Dataset):
    def __init__(self, split: str = "train", config: ToyConfig | None = None):
        self.config = config or ToyConfig()
        generator = torch.Generator().manual_seed(self.config.seed)
        records = []
        for message_id in range(self.config.num_messages):
            harmful = 1 if message_id >= self.config.num_messages // 2 else 0
            template = make_template(message_id, self.config.image_size)
            for index in range(self.config.samples_per_message):
                background = make_background(self.config.image_size, generator)
                strength = self.config.hidden_strength * (0.65 + 0.7 * torch.rand((), generator=generator).item())
                noise = torch.randn(3, self.config.image_size, self.config.image_size, generator=generator) * self.config.noise_std
                if index % 3 == 0:
                    hidden = torch.roll(template, shifts=(index % 7 - 3, index % 5 - 2), dims=(1, 2))
                elif index % 3 == 1:
                    hidden = torch.flip(template, dims=(2,))
                else:
                    hidden = template
                image = (background + strength * (hidden - 0.5) + noise).clamp(0, 1)
                records.append((image, message_id, harmful))
        perm = torch.randperm(len(records), generator=generator).tolist()
        cut_train = int(0.7 * len(perm))
        cut_val = int(0.85 * len(perm))
        if split == "train":
            chosen = perm[:cut_train]
        elif split == "val":
            chosen = perm[cut_train:cut_val]
        elif split == "test":
            chosen = perm[cut_val:]
        else:
            raise ValueError(f"unknown split: {split}")
        self.records = [records[i] for i in chosen]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, message_id, harmful = self.records[index]
        return {
            "image": image.float(),
            "message_id": torch.tensor(message_id, dtype=torch.long),
            "harmful": torch.tensor(harmful, dtype=torch.float32),
        }


def build_template_bank(config: ToyConfig | None = None) -> torch.Tensor:
    cfg = config or ToyConfig()
    return torch.stack([make_template(i, cfg.image_size) for i in range(cfg.num_messages)], dim=0).float()
