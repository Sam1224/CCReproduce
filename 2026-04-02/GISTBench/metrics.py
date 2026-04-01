from __future__ import annotations

import math
from typing import Iterable

from dataset import TAXONOMY


def interest_groundedness(*, predicted: list[str], ground_truth: list[str]) -> dict[str, float]:
    predicted_set = set(predicted)
    gt_set = set(ground_truth)

    tp = len(predicted_set & gt_set)
    precision = tp / max(1, len(predicted_set))
    recall = tp / max(1, len(gt_set))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def interest_specificity(*, predicted: Iterable[str]) -> float:
    """A proxy for Interest Specificity (IS).

    The paper measures how specific verified interests are in a taxonomy.
    Here we score a coarse interest as more specific when its taxonomy branch is
    narrower (fewer leaf categories).
    """

    scores: list[float] = []
    for interest in predicted:
        children = TAXONOMY["root"].get(interest)
        if not children:
            scores.append(0.0)
            continue
        scores.append(1.0 / math.log2(len(children) + 2))

    return float(sum(scores) / max(1, len(scores)))
