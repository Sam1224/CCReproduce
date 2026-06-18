from __future__ import annotations

"""Toy product-editing dataset for the ProductConsistency reproduction.

A "product" is a centered patch whose identity = (brand, text):
  * brand -> a base colour pattern in the product region (proxy for branding/logo)
  * text  -> TEXT_LEN cells, each a digit rendered as an intensity bar
             (proxy for on-product rendered text that OCR must read)

The background is one of N_STYLES styles. An *edit instruction* asks to re-render
the SAME product on a DIFFERENT background style. The SFT target is the ground-truth
edited image (new background, identical product region) — i.e. perfect identity
preservation. Description dicts ({"brand","text"}) are the interface shared with the
captioner / cyclic-consistency reward.
"""

import torch
from torch.utils.data import Dataset

from model import IMG, PROD, N_BRANDS, TEXT_LEN, N_CHARS, N_STYLES

_LO = (IMG - PROD) // 2
_HI = _LO + PROD


def _brand_color(brand: int) -> torch.Tensor:
    # deterministic distinct RGB per brand
    g = torch.Generator().manual_seed(1000 + brand)
    return torch.rand(3, generator=g) * 0.6 + 0.2


def _background(style: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7 + style)
    base = torch.rand(3, generator=g) * 0.5 + 0.25
    bg = base.view(3, 1, 1).expand(3, IMG, IMG).clone()
    # add a faint style-specific gradient so styles are visually distinct
    ramp = torch.linspace(0, 0.15, IMG).view(1, 1, IMG)
    return (bg + (style + 1) / N_STYLES * ramp).clamp(0, 1)


def render(brand: int, text: torch.Tensor, style: int) -> torch.Tensor:
    """Render product (brand,text) on background `style` -> [3, IMG, IMG]."""
    img = _background(style)
    col = _brand_color(brand).view(3, 1, 1)
    img[:, _LO:_HI, _LO:_HI] = col
    # render TEXT_LEN digit cells as vertical intensity bars in the bottom of patch
    cell_w = PROD // TEXT_LEN
    for i in range(TEXT_LEN):
        val = 0.15 + 0.85 * (int(text[i]) / (N_CHARS - 1))
        x0 = _LO + i * cell_w
        img[:, _HI - 4:_HI - 1, x0 + 1:x0 + cell_w - 1] = val
    return img.clamp(0, 1)


class ToyProductDataset(Dataset):
    def __init__(self, n: int = 512, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.brand = torch.randint(0, N_BRANDS, (n,), generator=g)
        self.text = torch.randint(0, N_CHARS, (n, TEXT_LEN), generator=g)
        self.src_style = torch.randint(0, N_STYLES, (n,), generator=g)
        # instruction = target background style (different from source)
        self.instr = (self.src_style + 1 + torch.randint(0, N_STYLES - 1, (n,), generator=g)) % N_STYLES

    def __len__(self):
        return self.brand.size(0)

    def __getitem__(self, i):
        src = render(int(self.brand[i]), self.text[i], int(self.src_style[i]))
        tgt = render(int(self.brand[i]), self.text[i], int(self.instr[i]))
        return {
            "src": src,
            "tgt": tgt,
            "instr": self.instr[i],
            "brand": self.brand[i],
            "text": self.text[i],
        }


def collate(batch):
    return {
        "src": torch.stack([b["src"] for b in batch]),
        "tgt": torch.stack([b["tgt"] for b in batch]),
        "instr": torch.stack([b["instr"] for b in batch]),
        "brand": torch.stack([b["brand"] for b in batch]),
        "text": torch.stack([b["text"] for b in batch]),
    }
