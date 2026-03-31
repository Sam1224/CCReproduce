from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


LABELS = ["L1", "L2", "L3", "L4"]
SUBSETS = ["general", "hard", "visual"]


@dataclass
class Example:
    subset: str
    q_brand: int
    q_cat: int
    q_color: int
    item_brand: int
    item_cat: int
    item_color_text: int
    item_color_img: int
    rule_ids: torch.Tensor  # (R,)
    y: int  # 0..3 for L1..L4


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def label_from_rules(q_cat: int, q_brand: int, q_color: int, item_cat: int, item_brand: int, item_color: int) -> int:
    # L1: category mismatch
    if item_cat != q_cat:
        return 0
    # L2: category ok but brand mismatch
    if item_brand != q_brand:
        return 1
    # L3: cat+brand ok but attribute mismatch
    if item_color != q_color:
        return 2
    # L4: perfect
    return 3


def make_dataset(n: int = 12000, n_cat: int = 40, n_brand: int = 120, n_color: int = 20, seed: int = 0) -> List[Example]:
    rng = _seed(seed)
    out: List[Example] = []

    for _ in range(n):
        subset = SUBSETS[int(rng.integers(0, len(SUBSETS)))]

        q_cat = int(rng.integers(0, n_cat))
        q_brand = int(rng.integers(0, n_brand))
        q_color = int(rng.integers(0, n_color))

        # build an item with controlled violation patterns
        if subset == "general":
            # mostly easy
            item_cat = q_cat if rng.random() < 0.85 else int(rng.integers(0, n_cat))
            item_brand = q_brand if rng.random() < 0.80 else int(rng.integers(0, n_brand))
            item_color_true = q_color if rng.random() < 0.70 else int(rng.integers(0, n_color))
        elif subset == "hard":
            # harder: more multi-constraint cases
            item_cat = q_cat if rng.random() < 0.70 else int(rng.integers(0, n_cat))
            item_brand = q_brand if rng.random() < 0.60 else int(rng.integers(0, n_brand))
            item_color_true = q_color if rng.random() < 0.55 else int(rng.integers(0, n_color))
        else:
            # visual salient: color is hidden from text and only in image
            item_cat = q_cat if rng.random() < 0.80 else int(rng.integers(0, n_cat))
            item_brand = q_brand if rng.random() < 0.70 else int(rng.integers(0, n_brand))
            item_color_true = q_color if rng.random() < 0.60 else int(rng.integers(0, n_color))

        # text may omit/blur color for visual subset
        item_color_text = item_color_true
        item_color_img = item_color_true
        if subset == "visual":
            item_color_text = int(rng.integers(0, n_color))  # noisy/wrong in text

        y = label_from_rules(q_cat, q_brand, q_color, item_cat, item_brand, item_color_true)

        # simple rule checklist id encoding
        # 0: category match, 1: brand match, 2: color match
        r0 = 1 if item_cat == q_cat else 0
        r1 = 1 if item_brand == q_brand else 0
        r2 = 1 if item_color_true == q_color else 0
        rule_ids = torch.tensor([r0, r1, r2], dtype=torch.long)

        out.append(
            Example(
                subset=subset,
                q_brand=q_brand,
                q_cat=q_cat,
                q_color=q_color,
                item_brand=item_brand,
                item_cat=item_cat,
                item_color_text=item_color_text,
                item_color_img=item_color_img,
                rule_ids=rule_ids,
                y=y,
            )
        )

    return out


def split(items: List[Example], frac: float = 0.85) -> Tuple[List[Example], List[Example]]:
    n = int(len(items) * frac)
    return items[:n], items[n:]
