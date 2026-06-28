from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class AnnotatedPair:
    qid: str
    pid: str
    true_grade: int
    pseudo_grade: int
    confidence: float
    bucket: str


def _clip_grade(g: int) -> int:
    return int(max(0, min(4, g)))


def noisy_grade(true_grade: int, bucket: str, level: str, rng: random.Random) -> Tuple[int, float]:
    """Simulate an annotator producing a 5-grade label with a confidence score."""

    # higher noise for hard buckets; lower noise for stronger models.
    bucket_noise = {
        "easy_pos": 0.10,
        "hard_pos": 0.22,
        "hard_neg": 0.20,
        "medium": 0.16,
        "easy_neg": 0.08,
        "": 0.15,
    }.get(bucket, 0.16)

    level_scale = {"fast": 1.35, "mid": 1.0, "strong": 0.65}.get(level, 1.0)
    sigma = bucket_noise * level_scale

    # translate sigma into a discrete perturbation
    delta = int(round(rng.gauss(0, sigma * 3.0)))
    pred = _clip_grade(true_grade + delta)

    # confidence reflects expected correctness
    conf = float(max(0.05, min(0.99, 1.0 - sigma * (abs(delta) + 0.5))))
    return pred, conf


class CalibratedCascade:
    """A minimal stand-in for the paper's calibrated model cascade.

    In the real system, this would be: cross-encoder -> small LLM -> larger LLM,
    with per-class isotonic calibration and deferral.

    Here we simulate that behavior by learning per-predicted-grade thresholds
    on a calibration set, then applying a 3-stage cascade.
    """

    def __init__(self, target_precision: float = 0.85, seed: int = 0):
        self.target_precision = target_precision
        self.rng = random.Random(seed)
        self.thresholds: Dict[int, float] = {g: 0.0 for g in range(5)}

    def fit(self, calibration: Sequence[Tuple[int, str]]) -> None:
        """calibration: (true_grade, bucket) pairs."""

        # For each predicted grade, estimate confidence threshold achieving target precision.
        by_pred: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
        for true_grade, bucket in calibration:
            pred, conf = noisy_grade(true_grade, bucket, "fast", self.rng)
            ok = pred == true_grade
            by_pred[pred].append((conf, ok))

        for pred_grade, items in by_pred.items():
            items.sort(key=lambda x: -x[0])
            tp = 0
            for i, (conf, ok) in enumerate(items, start=1):
                tp += 1 if ok else 0
                prec = tp / i
                if prec >= self.target_precision:
                    self.thresholds[pred_grade] = conf
            # if never reaches target, keep relatively high threshold
            if self.thresholds[pred_grade] == 0.0 and items:
                self.thresholds[pred_grade] = max(0.65, items[0][0] * 0.85)

    def annotate(self, true_grade: int, bucket: str) -> Tuple[int, float, str]:
        # stage 1 (fast)
        pred, conf = noisy_grade(true_grade, bucket, "fast", self.rng)
        if conf >= self.thresholds.get(pred, 0.7):
            return pred, conf, "fast"

        # stage 2 (mid)
        pred, conf = noisy_grade(true_grade, bucket, "mid", self.rng)
        if conf >= max(0.55, self.thresholds.get(pred, 0.6) - 0.08):
            return pred, conf, "mid"

        # stage 3 (strong)
        pred, conf = noisy_grade(true_grade, bucket, "strong", self.rng)
        return pred, conf, "strong"


def run_annotation_cascade(pairs: Sequence[Dict], seed: int = 0) -> List[AnnotatedPair]:
    rng = random.Random(seed)

    # Build a small calibration set from the provided pairs.
    cal = [(it["grade"], it.get("bucket", "")) for it in pairs]
    rng.shuffle(cal)
    cal = cal[: max(200, len(cal) // 8)]

    cascade = CalibratedCascade(target_precision=0.85, seed=seed)
    cascade.fit(cal)

    out: List[AnnotatedPair] = []
    for it in pairs:
        pred, conf, _stage = cascade.annotate(it["grade"], it.get("bucket", ""))
        out.append(
            AnnotatedPair(
                qid=it["qid"],
                pid=it["pid"],
                true_grade=int(it["grade"]),
                pseudo_grade=int(pred),
                confidence=float(conf),
                bucket=it.get("bucket", ""),
            )
        )

    return out
