import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.segmentation import slic


@dataclass(frozen=True)
class ToyFashionSpec:
    image_size: int = 64
    num_attrs: int = 3
    values_per_attr: int = 3


ATTR_NAMES = ["collar", "sleeve", "pattern"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw_base_shirt(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, fill) -> None:
    draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=fill)


def _draw_collar(draw: ImageDraw.ImageDraw, cx: int, cy: int, value: int, color) -> None:
    if value == 0:
        draw.polygon([(cx - 10, cy - 2), (cx + 10, cy - 2), (cx, cy + 12)], fill=color)
    elif value == 1:
        draw.ellipse([cx - 10, cy - 2, cx + 10, cy + 16], fill=color)
    else:
        draw.rectangle([cx - 10, cy - 2, cx + 10, cy + 8], fill=color)


def _draw_sleeve(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, value: int, color) -> None:
    mid_y = (y0 + y1) // 2
    if value == 0:
        draw.polygon([(x0, mid_y), (x0 - 12, mid_y + 6), (x0, mid_y + 16)], fill=color)
        draw.polygon([(x1, mid_y), (x1 + 12, mid_y + 6), (x1, mid_y + 16)], fill=color)
    elif value == 1:
        draw.polygon([(x0, mid_y), (x0 - 6, mid_y + 6), (x0, mid_y + 16)], fill=color)
        draw.polygon([(x1, mid_y), (x1 + 6, mid_y + 6), (x1, mid_y + 16)], fill=color)
    else:
        draw.polygon([(x0, mid_y), (x0 - 2, mid_y + 6), (x0, mid_y + 16)], fill=color)
        draw.polygon([(x1, mid_y), (x1 + 2, mid_y + 6), (x1, mid_y + 16)], fill=color)


def _draw_pattern(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, value: int, color) -> None:
    if value == 0:
        step = 8
        for x in range(x0, x1, step):
            draw.line([x, y0, x + (y1 - y0), y1], fill=color, width=2)
    elif value == 1:
        step = 7
        for x in range(x0, x1, step):
            draw.line([x, y0, x, y1], fill=color, width=2)
    else:
        step = 9
        for y in range(y0, y1, step):
            draw.line([x0, y, x1, y], fill=color, width=2)


def render_toy_fashion_image(
    attr_values: list[int],
    spec: ToyFashionSpec,
    rng: np.random.Generator,
) -> Image.Image:
    h = w = spec.image_size
    bg = (245, 247, 250)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    shirt_color = tuple(int(x) for x in rng.integers(80, 200, size=3))
    accent = tuple(int(x) for x in rng.integers(20, 80, size=3))

    x0, y0, x1, y1 = 14, 14, w - 14, h - 10
    _draw_base_shirt(draw, x0, y0, x1, y1, fill=shirt_color)

    collar_value, sleeve_value, pattern_value = attr_values
    _draw_collar(draw, (x0 + x1) // 2, y0 + 6, collar_value, color=accent)
    _draw_sleeve(draw, x0, y0, x1, y1, sleeve_value, color=accent)
    _draw_pattern(draw, x0 + 6, y0 + 18, x1 - 6, y1 - 6, pattern_value, color=accent)

    return img


class ToyFashionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_items: int,
        spec: ToyFashionSpec,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        rng = np.random.default_rng(seed)

        self.attr_values = rng.integers(
            low=0,
            high=spec.values_per_attr,
            size=(num_items, spec.num_attrs),
            dtype=np.int64,
        )
        self.images: list[Image.Image] = [
            render_toy_fashion_image(self.attr_values[i].tolist(), spec, rng)
            for i in range(num_items)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return {
            "image": self.images[idx],
            "attr_values": self.attr_values[idx].copy(),
        }


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tokenize_patch(img: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    c, h, w = img.shape
    ph = h // patch_size
    pw = w // patch_size

    patches = img[:, : ph * patch_size, : pw * patch_size].reshape(
        c, ph, patch_size, pw, patch_size
    )
    patches = patches.permute(1, 3, 0, 2, 4).contiguous().reshape(ph * pw, c, patch_size, patch_size)
    mean = patches.mean(dim=(2, 3))
    std = patches.std(dim=(2, 3))
    return torch.cat([mean, std], dim=1)


def tokenize_superpixel(img: torch.Tensor, n_segments: int = 48) -> torch.Tensor:
    c, h, w = img.shape
    np_img = img.permute(1, 2, 0).cpu().numpy()
    seg = slic(np_img, n_segments=n_segments, compactness=10.0, start_label=0)
    seg = seg.astype(np.int64)
    num_seg = int(seg.max()) + 1

    feats = []
    ys, xs = np.mgrid[0:h, 0:w]
    for k in range(num_seg):
        mask = seg == k
        if not np.any(mask):
            continue
        pix = np_img[mask]
        mean_rgb = pix.mean(axis=0)
        y_mean = ys[mask].mean() / float(h)
        x_mean = xs[mask].mean() / float(w)
        feats.append(np.concatenate([mean_rgb, [y_mean, x_mean]], axis=0))

    feats = np.stack(feats, axis=0).astype(np.float32)
    return torch.from_numpy(feats)


def tokenize_image(img: torch.Tensor, tokenizer: str) -> torch.Tensor:
    if tokenizer == "patch":
        return tokenize_patch(img)
    if tokenizer == "superpixel":
        return tokenize_superpixel(img)
    raise ValueError(f"Unknown tokenizer: {tokenizer}")


def split_indices(n: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(math.floor(n * train_ratio))
    return idx[:split], idx[split:]
