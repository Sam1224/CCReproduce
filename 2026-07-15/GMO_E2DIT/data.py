import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

OPS = ["add", "remove", "move", "replace", "exchange"]
VOCAB = {"<pad>": 0, "add": 1, "remove": 2, "move": 3, "replace": 4, "exchange": 5, "patch": 6, "sticker": 7, "left": 8, "right": 9, "top": 10, "bottom": 11, "red": 12, "green": 13, "blue": 14, "yellow": 15}
COLORS = {
    "red": torch.tensor([1.0, 0.15, 0.12]),
    "green": torch.tensor([0.1, 0.8, 0.25]),
    "blue": torch.tensor([0.12, 0.35, 1.0]),
    "yellow": torch.tensor([1.0, 0.85, 0.1]),
}


@dataclass
class EditExample:
    image: torch.Tensor
    target: torch.Tensor
    instruction_ids: torch.Tensor
    op_id: int
    source_mask: torch.Tensor
    target_mask: torch.Tensor
    success: float


def _box_mask(size: int, row: int, col: int, box: int = 4) -> torch.Tensor:
    mask = torch.zeros(1, size, size)
    mask[:, row : row + box, col : col + box] = 1.0
    return mask


def _paint(image: torch.Tensor, mask: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    return image * (1.0 - mask) + color.view(3, 1, 1) * mask


def tokenize(text: str, max_len: int = 10) -> torch.Tensor:
    ids = [VOCAB.get(tok, 0) for tok in text.lower().replace("-", " ").split()]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class ToyEcommerceEditDataset(Dataset):
    def __init__(self, n: int = 512, image_size: int = 16, seed: int = 7):
        self.n = n
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.positions = [(2, 2, "top left"), (2, 10, "top right"), (10, 2, "bottom left"), (10, 10, "bottom right")]
        self.samples = [self._make_one() for _ in range(n)]

    def _base_image(self) -> Tuple[torch.Tensor, List[Tuple[int, int, str, str]]]:
        image = torch.ones(3, self.image_size, self.image_size) * 0.92
        patches = []
        for row, col, name in self.positions:
            color_name = self.rng.choice(list(COLORS))
            image = _paint(image, _box_mask(self.image_size, row, col), COLORS[color_name])
            patches.append((row, col, name, color_name))
        return image, patches

    def _make_one(self) -> EditExample:
        image, patches = self._base_image()
        op = self.rng.choice(OPS)
        src = self.rng.choice(patches)
        dst = self.rng.choice([p for p in patches if p[:2] != src[:2]])
        new_color = self.rng.choice([c for c in COLORS if c != src[3]])
        source_mask = _box_mask(self.image_size, src[0], src[1])
        target_mask = _box_mask(self.image_size, dst[0], dst[1])
        target = image.clone()

        if op == "remove":
            target = target * (1.0 - source_mask) + 0.92 * source_mask
            text = f"remove {src[2]} sticker"
        elif op == "add":
            target = _paint(target, target_mask, COLORS[new_color])
            text = f"add {new_color} patch {dst[2]}"
        elif op == "replace":
            target = _paint(target, source_mask, COLORS[new_color])
            text = f"replace {src[3]} patch {new_color}"
        elif op == "move":
            patch_pixels = target * source_mask
            target = target * (1.0 - source_mask) + 0.92 * source_mask
            target = target * (1.0 - target_mask) + patch_pixels.sum(dim=(1, 2), keepdim=True) / source_mask.sum() * target_mask
            text = f"move {src[2]} patch {dst[2]}"
        else:
            src_pixels = target * source_mask
            dst_pixels = target * target_mask
            target = target * (1.0 - source_mask - target_mask) + src_pixels.sum(dim=(1, 2), keepdim=True) / source_mask.sum() * target_mask + dst_pixels.sum(dim=(1, 2), keepdim=True) / target_mask.sum() * source_mask
            text = f"exchange {src[2]} {dst[2]} patch"

        return EditExample(image, target, tokenize(text), OPS.index(op), source_mask, target_mask, 1.0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.samples[idx]
        return {
            "image": ex.image,
            "target": ex.target,
            "instruction_ids": ex.instruction_ids,
            "op_id": torch.tensor(ex.op_id, dtype=torch.long),
            "source_mask": ex.source_mask,
            "target_mask": ex.target_mask,
            "success": torch.tensor([ex.success], dtype=torch.float32),
        }


def make_loaders(batch_size: int = 32):
    train = ToyEcommerceEditDataset(768, seed=11)
    test = ToyEcommerceEditDataset(160, seed=23)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
    )
