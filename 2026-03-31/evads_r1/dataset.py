from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

TASKS = ["BP", "CM", "ML", "CI", "RC"]


@dataclass
class Sample:
    task: str
    video: torch.Tensor  # (T, d_v)
    asr: torch.Tensor  # (L,)
    ocr: torch.Tensor  # (L2,)
    q: torch.Tensor  # (Lq,)
    y: int


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_dataset(n: int = 6000, d_v: int = 48, t: int = 12, vocab: int = 1000, n_classes: int = 12, seed: int = 0) -> List[Sample]:
    rng = _seed(seed)

    # latent factors for "commercial intent"
    z = rng.normal(size=(n, 16)).astype(np.float32)

    out: List[Sample] = []
    for i in range(n):
        task = TASKS[int(rng.integers(0, len(TASKS)))]

        # "dense" video: higher variance for e-commerce ads
        density = 1.0 + (task in ("ML", "CI", "RC")) * 0.6
        video = rng.normal(scale=density, size=(t, d_v)).astype(np.float32)

        # ASR/OCR token bags
        asr_len = int(rng.integers(8, 22))
        ocr_len = int(rng.integers(4, 16))
        q_len = int(rng.integers(6, 18))
        asr = rng.integers(0, vocab, size=(asr_len,), dtype=np.int64)
        ocr = rng.integers(0, vocab, size=(ocr_len,), dtype=np.int64)
        q = rng.integers(0, vocab, size=(q_len,), dtype=np.int64)

        # label rule: combine latent + task
        y = int((np.argmax(z[i]) + TASKS.index(task)) % n_classes)

        out.append(
            Sample(
                task=task,
                video=torch.tensor(video),
                asr=torch.tensor(asr),
                ocr=torch.tensor(ocr),
                q=torch.tensor(q),
                y=y,
            )
        )

    return out


def split(items: List[Sample], frac: float = 0.85) -> Tuple[List[Sample], List[Sample]]:
    n = int(len(items) * frac)
    return items[:n], items[n:]
