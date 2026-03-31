from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

SCENES = [
    "illegal",
    "false_marketing",
    "misleading_ops",
    "privacy",
    "superstition",
    "finance",
    "medical",
]

TYPES = [
    "income_exaggeration",
    "claim_without_proof",
    "bait_click",
    "privacy_leak",
    "illegal_goods",
]


@dataclass
class AdExample:
    frames: torch.Tensor  # (N, d)
    asr: torch.Tensor  # (L,)
    label_scene: int
    label_type: int
    label_risky: int


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_dataset(n: int = 8000, n_frames: int = 16, d: int = 64, vocab: int = 800, seed: int = 0) -> List[AdExample]:
    rng = _seed(seed)
    out: List[AdExample] = []

    # risk prompts simulated as fixed vectors
    prompt_vecs = rng.normal(size=(len(SCENES), d)).astype(np.float32)

    for _ in range(n):
        label_scene = int(rng.integers(0, len(SCENES)))
        label_type = int(rng.integers(0, len(TYPES)))
        risky = 1 if rng.random() < 0.35 else 0

        base = rng.normal(size=(d,)).astype(np.float32)
        if risky:
            base = base + 0.8 * prompt_vecs[label_scene]

        frames = base + rng.normal(scale=0.8, size=(n_frames, d)).astype(np.float32)

        # ASR tokens correlated with type when risky
        L = int(rng.integers(12, 28))
        asr = rng.integers(0, vocab, size=(L,), dtype=np.int64)
        if risky:
            asr[:3] = np.array([label_type * 3 + 1, label_type * 3 + 2, label_type * 3 + 3], dtype=np.int64) % vocab

        out.append(
            AdExample(
                frames=torch.tensor(frames),
                asr=torch.tensor(asr),
                label_scene=label_scene,
                label_type=label_type,
                label_risky=risky,
            )
        )

    return out


def split(items: List[AdExample], frac: float = 0.85) -> Tuple[List[AdExample], List[AdExample]]:
    n = int(len(items) * frac)
    return items[:n], items[n:]
