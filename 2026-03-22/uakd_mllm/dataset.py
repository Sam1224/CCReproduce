from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ToyMMBatch:
    img: torch.Tensor  # (N, D_img)
    txt: torch.Tensor  # (N, L)
    y: torch.Tensor  # (N,)
    ambiguous: torch.Tensor  # (N,) bool


def make_dataset(n: int = 2000, seed: int = 1, num_classes: int = 5, txt_vocab: int = 128) -> ToyMMBatch:
    g = torch.Generator().manual_seed(seed)

    d_img = 32
    L = 16

    y = torch.randint(0, num_classes, (n,), generator=g)

    # Class prototypes for image/text.
    img_proto = torch.randn(num_classes, d_img, generator=g)
    txt_proto = torch.randn(num_classes, L, generator=g)

    # Build “clean” features.
    img = img_proto[y] + 0.6 * torch.randn(n, d_img, generator=g)
    txt_dense = txt_proto[y] + 0.6 * torch.randn(n, L, generator=g)

    # Ambiguous samples: mix another class.
    ambiguous = torch.rand(n, generator=g) < 0.25
    alt = torch.randint(0, num_classes, (n,), generator=g)
    mix = 0.5
    img[ambiguous] = mix * img[ambiguous] + (1 - mix) * img_proto[alt[ambiguous]]
    txt_dense[ambiguous] = mix * txt_dense[ambiguous] + (1 - mix) * txt_proto[alt[ambiguous]]

    # Quantize text to fake token IDs.
    txt = ((txt_dense - txt_dense.min()) / (txt_dense.max() - txt_dense.min() + 1e-6) * (txt_vocab - 1)).long()

    return ToyMMBatch(img=img, txt=txt, y=y, ambiguous=ambiguous)


def split(batch: ToyMMBatch, frac: float = 0.8) -> Tuple[ToyMMBatch, ToyMMBatch]:
    n = batch.y.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)

    def sel(i: torch.Tensor) -> ToyMMBatch:
        return ToyMMBatch(
            img=batch.img[i],
            txt=batch.txt[i],
            y=batch.y[i],
            ambiguous=batch.ambiguous[i],
        )

    return sel(idx[:k]), sel(idx[k:])
