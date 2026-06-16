"""Toy dataset for LazyMCoT.

Synthetic "high-res" images, each containing one *small* saturated target object
on a gray background cluttered with larger gray distractor blobs. A multiple-choice
question asks for the color of the small target. Because the target is small, a base
VLM that perceives the image at a coarse token grid (heavy downsampling) loses the
target signal -> low first-token confidence -> "hard" sample that benefits from
Collaborative Grounding (focusing / zooming on the target). Larger / higher-contrast
targets are "easy" and answered correctly in a single pass.

This mirrors the real setting (V* Bench / HR-Bench: tiny objects in high-res images)
while staying tiny and CPU-friendly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# Vocabulary of color words (V). Chosen to be mutually far in RGB space so that
# correctness is driven by *target visibility*, not color confusion.
# ----------------------------------------------------------------------------
VOCAB: List[str] = ["red", "green", "blue", "yellow", "magenta", "cyan"]
COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "red": (230, 25, 25),
    "green": (25, 160, 25),
    "blue": (25, 25, 230),
    "yellow": (235, 235, 20),
    "magenta": (230, 25, 230),
    "cyan": (20, 210, 210),
}
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
COLOR_MATRIX = np.array([COLOR_RGB[w] for w in VOCAB], dtype=np.float64) / 255.0  # [V,3]


@dataclass
class ToyConfig:
    img_size: int = 384            # simulated "high-res" canvas
    n_options: int = 4             # K candidate options per question
    n_distractors: int = 6         # gray clutter blobs
    target_min: int = 7            # smallest target diameter (px) -> hard
    target_max: int = 28           # largest target diameter (px)  -> easy
    distractor_min: int = 34
    distractor_max: int = 80
    bg_gray: int = 55
    n_cal: int = 320               # routing-calibration split size
    n_test: int = 140              # test split size


@dataclass
class Sample:
    image: Image.Image
    question: str
    options: List[str]
    option_ids: List[int]
    answer: str
    answer_id: int
    target_color: str
    target_box: Tuple[int, int, int, int]
    target_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _rand_gray(rng: random.Random) -> Tuple[int, int, int]:
    v = rng.randint(25, 115)
    j = rng.randint(-6, 6)
    return (max(0, min(255, v + j)),) * 3


def _make_image(cfg: ToyConfig, rng: random.Random) -> Tuple[Image.Image, str, Tuple[int, int, int, int], int]:
    S = cfg.img_size
    img = Image.new("RGB", (S, S), (cfg.bg_gray,) * 3)
    d = ImageDraw.Draw(img)

    # gray distractor clutter (high saturation = 0, so it never excites color logits)
    for _ in range(cfg.n_distractors):
        r = rng.randint(cfg.distractor_min, cfg.distractor_max)
        x = rng.randint(0, S - r)
        y = rng.randint(0, S - r)
        d.ellipse([x, y, x + r, y + r], fill=_rand_gray(rng))

    # one small saturated target in a clear-ish spot
    color = rng.choice(VOCAB)
    size = rng.randint(cfg.target_min, cfg.target_max)
    tx = rng.randint(size, S - 2 * size)
    ty = rng.randint(size, S - 2 * size)
    box = (tx, ty, tx + size, ty + size)
    d.ellipse(list(box), fill=COLOR_RGB[color])
    return img, color, box, size


def make_sample(cfg: ToyConfig, rng: random.Random) -> Sample:
    img, color, box, size = _make_image(cfg, rng)
    # build K options including the true color
    others = [w for w in VOCAB if w != color]
    rng.shuffle(others)
    options = [color] + others[: cfg.n_options - 1]
    rng.shuffle(options)
    option_ids = [WORD2ID[w] for w in options]
    q = "What is the color of the small target object in the image?"
    return Sample(
        image=img, question=q, options=options, option_ids=option_ids,
        answer=color, answer_id=WORD2ID[color], target_color=color,
        target_box=box, target_size=size,
    )


class ToyVGDataset:
    """Indexable dataset of `Sample`s; interface matches model + scripts."""

    def __init__(self, cfg: ToyConfig, n: int, seed: int):
        self.cfg = cfg
        rng = random.Random(seed)
        self.samples: List[Sample] = [make_sample(cfg, rng) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Sample:
        return self.samples[i]

    def __iter__(self):
        return iter(self.samples)


def build_splits(cfg: ToyConfig, seed: int = 0) -> Tuple[ToyVGDataset, ToyVGDataset]:
    """Held-out routing-calibration set D_cal and the test set."""
    cal = ToyVGDataset(cfg, cfg.n_cal, seed=seed + 1)
    test = ToyVGDataset(cfg, cfg.n_test, seed=seed + 9999)
    return cal, test


def collect_routing_dataset(dataset: ToyVGDataset, vlm) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Produce D_cal features: x=(topp, delta_logit) and label y in {0,1}.

    y = 0  if the single-pass (Direct) answer is correct   (ori-correct)
    y = 1  if the single-pass answer is wrong               (ori-wrong, must-recall)
    """
    from model import compute_first_token_stats  # lazy import to avoid cycle

    X, y = [], []
    for s in dataset:
        logits = vlm.first_token_logits(s.image, s.question, s.option_ids)
        topp, delta = compute_first_token_stats(logits, s.option_ids)
        pred = s.option_ids[int(np.argmax(logits[s.option_ids]))]
        X.append([topp, delta])
        y.append(0 if pred == s.answer_id else 1)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    info = {"n": len(y), "n_wrong": int(y.sum()), "ori_acc": float((y == 0).mean())}
    return X, y, info
