import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


@dataclass
class FashionDetailSample:
    reference: torch.Tensor
    focus_mask: torch.Tensor
    target_detail: torch.Tensor
    category_id: torch.Tensor


class ToyFashionDetailDataset(Dataset):
    """Toy interface for FDBench-style reference/detail pairs."""

    def __init__(self, length: int = 256, image_size: int = 64, detail_size: int = 32, categories: int = 41):
        self.length = length
        self.image_size = image_size
        self.detail_size = detail_size
        self.categories = categories

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        category_id = torch.tensor(index % self.categories, dtype=torch.long)
        reference = torch.rand(3, self.image_size, self.image_size, generator=generator)
        patch_size = random.Random(index).choice([14, 16, 18, 20, 22])
        top = random.Random(index + 17).randint(0, self.image_size - patch_size)
        left = random.Random(index + 31).randint(0, self.image_size - patch_size)
        focus_mask = torch.zeros(1, self.image_size, self.image_size)
        focus_mask[:, top : top + patch_size, left : left + patch_size] = 1.0
        crop = reference[:, top : top + patch_size, left : left + patch_size].unsqueeze(0)
        target_detail = F.interpolate(crop, size=(self.detail_size, self.detail_size), mode="bilinear", align_corners=False).squeeze(0)
        texture = torch.sin(torch.linspace(0, 12, self.detail_size)).view(1, 1, -1)
        target_detail = (0.82 * target_detail + 0.18 * texture).clamp(0, 1)
        return {
            "reference": reference,
            "focus_mask": focus_mask,
            "target_detail": target_detail,
            "category_id": category_id,
        }
