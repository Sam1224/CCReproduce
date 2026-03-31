from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch


DIGITS_3x5 = {
    0: [
        "###",
        "# #",
        "# #",
        "# #",
        "###",
    ],
    1: [
        " ##",
        "  #",
        "  #",
        "  #",
        " ###",
    ],
    2: [
        "###",
        "  #",
        "###",
        "#  ",
        "###",
    ],
    3: [
        "###",
        "  #",
        " ##",
        "  #",
        "###",
    ],
    4: [
        "# #",
        "# #",
        "###",
        "  #",
        "  #",
    ],
    5: [
        "###",
        "#  ",
        "###",
        "  #",
        "###",
    ],
    6: [
        "###",
        "#  ",
        "###",
        "# #",
        "###",
    ],
    7: [
        "###",
        "  #",
        "  #",
        "  #",
        "  #",
    ],
    8: [
        "###",
        "# #",
        "###",
        "# #",
        "###",
    ],
    9: [
        "###",
        "# #",
        "###",
        "  #",
        "###",
    ],
}


def _render_digit(img: np.ndarray, *, digit: int, top: int, left: int, scale: int) -> None:
    pattern = DIGITS_3x5[int(digit)]
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            if ch != "#":
                continue
            rr0 = top + r * scale
            cc0 = left + c * scale
            img[rr0 : rr0 + scale, cc0 : cc0 + scale] = 1.0


def _downsample_mean(img: np.ndarray, out_hw: int) -> np.ndarray:
    h, w = img.shape
    assert h == w
    factor = h // out_hw
    assert h % out_hw == 0
    x = img.reshape(out_hw, factor, out_hw, factor).mean(axis=(1, 3))
    return x


@dataclass
class Crop:
    y0: int
    x0: int
    y1: int
    x1: int


def make_3x3_crops(*, image_size: int = 64, crop_size: int = 32) -> List[Crop]:
    # 3x3 grid with overlaps. Positions: 0, 16, 32 for crop_size=32.
    step = (image_size - crop_size) // 2
    offsets = [0, step, image_size - crop_size]
    crops: List[Crop] = []
    for y0 in offsets:
        for x0 in offsets:
            crops.append(Crop(y0=y0, x0=x0, y1=y0 + crop_size, x1=x0 + crop_size))
    return crops


@dataclass
class Example:
    img_high: np.ndarray  # [64,64]
    img_low: np.ndarray  # [16,16]
    crop_id: int  # 0..8
    need_crop: bool
    label: int  # 0..9


def make_example(*, rng: np.random.Generator, image_size: int = 64) -> Example:
    img = np.zeros((image_size, image_size), dtype=np.float32)

    digit = int(rng.integers(0, 10))
    need_crop = bool(rng.random() < 0.6)

    # When crop is needed, use a tiny stamp that is hard to see after downsampling.
    scale = 1 if need_crop else 3

    crops = make_3x3_crops(image_size=image_size, crop_size=32)
    crop_id = int(rng.integers(0, len(crops)))
    crop = crops[crop_id]

    # Place the digit inside the chosen crop.
    # The digit is 3x5 scaled => width=3*scale, height=5*scale.
    dh, dw = 5 * scale, 3 * scale
    top = int(rng.integers(crop.y0, max(crop.y0 + 1, crop.y1 - dh)))
    left = int(rng.integers(crop.x0, max(crop.x0 + 1, crop.x1 - dw)))
    _render_digit(img, digit=digit, top=top, left=left, scale=scale)

    img_low = _downsample_mean(img, 16)
    return Example(img_high=img, img_low=img_low, crop_id=crop_id, need_crop=need_crop, label=digit)


def get_crop_lowres(img_high: np.ndarray, *, crop_id: int) -> np.ndarray:
    crop = make_3x3_crops(image_size=img_high.shape[0], crop_size=32)[int(crop_id)]
    patch = img_high[crop.y0 : crop.y1, crop.x0 : crop.x1]
    return _downsample_mean(patch, 16)


class AwaResToyDataset:
    def __init__(self, *, n_samples: int, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._examples = [make_example(rng=self._rng) for _ in range(n_samples)]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


def get_crop_lowres_torch(img_high: torch.Tensor, *, crop_id: torch.Tensor) -> torch.Tensor:
    # img_high: [B,1,64,64], crop_id: [B] in 0..8
    crops = make_3x3_crops(image_size=img_high.shape[-1], crop_size=32)
    out: List[torch.Tensor] = []
    for i in range(img_high.shape[0]):
        c = crops[int(crop_id[i].item())]
        patch = img_high[i, :, c.y0 : c.y1, c.x0 : c.x1].contiguous()  # [1,32,32]
        patch = patch.view(1, 16, 2, 16, 2).mean(dim=(2, 4))  # [1,16,16]
        out.append(patch)
    return torch.stack(out, dim=0)


@dataclass
class Batch:
    img_high: torch.Tensor  # [B,1,64,64]
    img_low: torch.Tensor  # [B,1,16,16]
    crop_id: torch.Tensor  # [B] oracle crop index 0..8
    img_crop: torch.Tensor  # [B,1,16,16] oracle crop
    action: torch.Tensor  # [B] 0=no_crop, 1..9=crop+1
    label: torch.Tensor  # [B] digit


def collate_sft(examples: List[Example]) -> Batch:
    img_high = torch.from_numpy(np.stack([e.img_high for e in examples], axis=0)).unsqueeze(1)
    img_low = torch.from_numpy(np.stack([e.img_low for e in examples], axis=0)).unsqueeze(1)
    crop_id = torch.tensor([e.crop_id for e in examples]).long()
    img_crop = get_crop_lowres_torch(img_high, crop_id=crop_id)

    action = torch.tensor([0 if not e.need_crop else (e.crop_id + 1) for e in examples]).long()
    label = torch.tensor([e.label for e in examples]).long()

    return Batch(img_high=img_high, img_low=img_low, crop_id=crop_id, img_crop=img_crop, action=action, label=label)


def batch_iter_sft(ds: AwaResToyDataset, *, batch_size: int, seed: int = 0) -> Iterator[Batch]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            chunk = idx[i : i + batch_size]
            yield collate_sft([ds[int(j)] for j in chunk])
