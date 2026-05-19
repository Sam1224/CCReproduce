from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


VOCAB = {
    "<pad>": 0,
    "red": 1,
    "green": 2,
    "blue": 3,
    "yellow": 4,
    "square": 5,
    "circle": 6,
}

ID2TOKEN = {v: k for k, v in VOCAB.items()}


@dataclass(frozen=True)
class ToyItem:
    image: torch.Tensor  # (3, H, W), float32 in [0,1]
    text_ids: torch.Tensor  # (L,)
    box_xyxy: Tuple[int, int, int, int]  # pixel box on full item image


def _draw_square(canvas: np.ndarray, xyxy: Tuple[int, int, int, int], rgb: Tuple[float, float, float]) -> None:
    x1, y1, x2, y2 = xyxy
    canvas[:, y1:y2, x1:x2] = np.asarray(rgb, dtype=np.float32)[:, None, None]


def _draw_circle(canvas: np.ndarray, xyxy: Tuple[int, int, int, int], rgb: Tuple[float, float, float]) -> None:
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = min(x2 - x1, y2 - y1) / 2

    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    patch = canvas[:, y1:y2, x1:x2]
    patch[:, mask] = np.asarray(rgb, dtype=np.float32)[:, None]


def _color_rgb(name: str) -> Tuple[float, float, float]:
    return {
        "red": (0.95, 0.25, 0.25),
        "green": (0.20, 0.75, 0.35),
        "blue": (0.25, 0.45, 0.95),
        "yellow": (0.95, 0.85, 0.25),
    }[name]


def _sample_box(rng: random.Random, canvas_size: int, obj_size: int) -> Tuple[int, int, int, int]:
    x1 = rng.randint(0, canvas_size - obj_size)
    y1 = rng.randint(0, canvas_size - obj_size)
    return (x1, y1, x1 + obj_size, y1 + obj_size)


def make_toy_item(rng: random.Random, image_size: int = 64, obj_size: int = 18) -> ToyItem:
    color = rng.choice(["red", "green", "blue", "yellow"])
    shape = rng.choice(["square", "circle"])

    canvas = np.zeros((3, image_size, image_size), dtype=np.float32)

    box = _sample_box(rng, image_size, obj_size)
    rgb = _color_rgb(color)

    if shape == "square":
        _draw_square(canvas, box, rgb)
    else:
        _draw_circle(canvas, box, rgb)

    # add 1-2 distractors
    for _ in range(rng.randint(1, 2)):
        d_color = rng.choice([c for c in ["red", "green", "blue", "yellow"] if c != color])
        d_shape = rng.choice(["square", "circle"])
        d_box = _sample_box(rng, image_size, obj_size)
        if d_shape == "square":
            _draw_square(canvas, d_box, _color_rgb(d_color))
        else:
            _draw_circle(canvas, d_box, _color_rgb(d_color))

    text_ids = torch.tensor([VOCAB[color], VOCAB[shape]], dtype=torch.long)

    return ToyItem(image=torch.from_numpy(canvas), text_ids=text_ids, box_xyxy=box)


def crop_query_from_item(item: ToyItem, query_size: int = 32) -> torch.Tensor:
    x1, y1, x2, y2 = item.box_xyxy
    crop = item.image[:, y1:y2, x1:x2].unsqueeze(0)  # (1,3,h,w)
    crop = F.interpolate(crop, size=(query_size, query_size), mode="bilinear", align_corners=False)
    return crop.squeeze(0)


def box_to_patch_mask(
    box_xyxy: Tuple[int, int, int, int],
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Return (num_patches,) mask in {0,1}."""

    x1, y1, x2, y2 = box_xyxy
    grid = image_size // patch_size
    mask = torch.zeros((grid, grid), dtype=torch.float32)

    px1 = int(math.floor(x1 / patch_size))
    py1 = int(math.floor(y1 / patch_size))
    px2 = int(math.ceil(x2 / patch_size))
    py2 = int(math.ceil(y2 / patch_size))

    mask[py1:py2, px1:px2] = 1.0
    return mask.flatten()


class ToyIMMRDataset(torch.utils.data.Dataset):
    """Pairs (query crop, full item image + text) with supervision box for distillation."""

    def __init__(self, size: int = 6000, seed: int = 0) -> None:
        self._size = size
        self._seed = seed

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self._seed + idx)
        item = make_toy_item(rng)
        query = crop_query_from_item(item)
        box_mask = box_to_patch_mask(item.box_xyxy, image_size=item.image.shape[-1], patch_size=8)

        return {
            "query_image": query,
            "item_image": item.image,
            "item_text_ids": item.text_ids,
            "item_box_mask": box_mask,
        }


def collate_immr(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    query = torch.stack([b["query_image"] for b in batch], dim=0)
    item = torch.stack([b["item_image"] for b in batch], dim=0)
    box_mask = torch.stack([b["item_box_mask"] for b in batch], dim=0)

    text = torch.stack([b["item_text_ids"] for b in batch], dim=0)

    return {
        "query_image": query,
        "item_image": item,
        "item_text": text,
        "item_box_mask": box_mask,
    }


class ToyIMMRGallery:
    def __init__(self, num_items: int, seed: int = 123) -> None:
        self.items: List[ToyItem] = []
        rng = random.Random(seed)
        for _ in range(num_items):
            self.items.append(make_toy_item(rng))


class ToyIMMRQueryDataset(torch.utils.data.Dataset):
    def __init__(self, gallery: ToyIMMRGallery, num_queries: int, seed: int = 999) -> None:
        self.gallery = gallery
        rng = random.Random(seed)
        self.targets = [rng.randrange(0, len(gallery.items)) for _ in range(num_queries)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target = self.targets[idx]
        item = self.gallery.items[target]
        query = crop_query_from_item(item)
        return {"query_image": query, "target": torch.tensor(target, dtype=torch.long)}


class ToyIMMRItemDataset(torch.utils.data.Dataset):
    def __init__(self, gallery: ToyIMMRGallery) -> None:
        self.gallery = gallery

    def __len__(self) -> int:
        return len(self.gallery.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.gallery.items[idx]
        return {"item_image": item.image, "item_text": item.text_ids}


def collate_query(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "query_image": torch.stack([b["query_image"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
    }


def collate_item(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "item_image": torch.stack([b["item_image"] for b in batch], dim=0),
        "item_text": torch.stack([b["item_text"] for b in batch], dim=0),
    }
