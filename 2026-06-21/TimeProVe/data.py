"""Toy synthetic dataset for TimeProVe (offline, CPU).

Mirrors the paper's LVQA setting at toy scale:
  * a long (untrimmed) ADL video -> a sequence of T segment features v in R^{T x D}
    (paper: each 16-frame segment encoded by a frozen I3D/CLIP backbone),
  * a ground-truth action timeline A = {(c_i, s_i, e_i)} used both to supervise the
    Action Detector and to build query-conditioned QA pairs,
  * open-ended QA pairs with a temporal intent tau (BEFORE/AFTER/FIRST/LAST/STATE...),
    a content-token set Q(q), a gold answer, and a gold supporting window.

Everything is deterministic given a seed so train/test stay aligned.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

# ---- toy ADL action vocabulary ----------------------------------------------
ACTIONS = [
    "cook", "drink", "read", "walk", "clean",
    "eat", "sit", "wash_hands", "take_medicine", "use_phone",
]
NUM_CLASSES = len(ACTIONS)
ACT2ID = {a: i for i, a in enumerate(ACTIONS)}

# answer / content-token vocabulary (action labels + yes/no for existence)
ANSWERS = ACTIONS + ["yes", "no"]
TOK2ID = {t: i for i, t in enumerate(ANSWERS + ["<unk>"])}
VOCAB = len(TOK2ID)
UNK = TOK2ID["<unk>"]

D = 24                       # segment feature dim (toy stand-in for I3D/CLIP)
T = 48                       # number of temporal segments per video (video length L)


def tok_id(t: str) -> int:
    return TOK2ID.get(t, UNK)


def content_tokens(action: str) -> List[str]:
    """T(a): normalized content-token set of an action label (Sec 3.1.2)."""
    return [action]


@dataclass
class QASample:
    feats: torch.Tensor                 # [T, D] segment features
    seg_labels: torch.Tensor            # [T, |C|] multi-hot gt action activations
    events: List[Tuple[int, int, int]]  # gold timeline (c, s, e), inclusive segments
    question: str
    q_tokens: List[str]                 # Q(q): query content tokens
    tau: str                            # temporal intent
    gold_answer: str
    gold_window: Optional[Tuple[int, int]]
    L: int = T


def _prototypes(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_CLASSES, D, generator=g)


PROTO = _prototypes()


def _make_video(rng: random.Random, n_events: int):
    """Plant n_events into a T-segment video; return (feats, seg_labels, events)."""
    seg_labels = torch.zeros(T, NUM_CLASSES)
    events: List[Tuple[int, int, int]] = []
    used_classes: set = set()
    for _ in range(n_events):
        c = rng.randrange(NUM_CLASSES)
        dur = rng.randint(2, 6)
        s = rng.randint(0, T - dur - 1)
        e = s + dur - 1
        for t in range(s, e + 1):
            seg_labels[t, c] = 1.0
        events.append((c, s, e))
        used_classes.add(c)
    # features: sum of active prototypes (normalized) + background noise
    feats = torch.zeros(T, D)
    for t in range(T):
        active = seg_labels[t].nonzero(as_tuple=True)[0]
        if len(active):
            feats[t] = PROTO[active].mean(0)
    feats = feats + 0.30 * torch.randn(T, D)
    events.sort(key=lambda x: x[1])
    return feats, seg_labels, events


def _questions(rng: random.Random, feats, seg_labels, events) -> List[QASample]:
    qs: List[QASample] = []
    present = sorted({c for c, _, _ in events})
    absent = [c for c in range(NUM_CLASSES) if c not in present]

    def mk(question, q_tokens, tau, ans, win):
        return QASample(feats, seg_labels, events, question, q_tokens, tau, ans, win)

    # 1) existence (yes): an action that occurs
    if present:
        c = rng.choice(present)
        ev = next(e for e in events if e[0] == c)
        qs.append(mk(f"Did the person {ACTIONS[c]}?", [ACTIONS[c]], "STATE",
                     "yes", (ev[1], ev[2])))
    # 2) existence (no): an action that does not occur
    if absent:
        c = rng.choice(absent)
        qs.append(mk(f"Did the person {ACTIONS[c]}?", [ACTIONS[c]], "STATE",
                     "no", None))
    # 3) first action
    first = events[0]
    qs.append(mk("What did the person do first?", [], "FIRST",
                 ACTIONS[first[0]], (first[1], first[2])))
    # 4) last action
    last = max(events, key=lambda x: x[2])
    qs.append(mk("What did the person do last?", [], "LAST",
                 ACTIONS[last[0]], (last[1], last[2])))
    # 5) after-X action
    for i, pivot in enumerate(events):
        nxt = next((e for e in events if e[1] > pivot[2]), None)
        if nxt is not None and nxt[0] != pivot[0]:
            qs.append(mk(f"What did the person do after {ACTIONS[pivot[0]]}?",
                         [ACTIONS[pivot[0]]], "AFTER",
                         ACTIONS[nxt[0]], (nxt[1], nxt[2])))
            break
    return qs


def make_dataset(n_videos: int = 40, seed: int = 0) -> List[QASample]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    data: List[QASample] = []
    for _ in range(n_videos):
        n_events = rng.randint(4, 7)
        feats, seg_labels, events = _make_video(rng, n_events)
        data.extend(_questions(rng, feats, seg_labels, events))
    return data


def split(n_train_videos: int = 40, n_test_videos: int = 16):
    train = make_dataset(n_train_videos, seed=0)
    test = make_dataset(n_test_videos, seed=123)
    return train, test
