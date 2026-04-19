from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class QAExample:
    qid: int
    question: str
    context: str
    answer: str
    is_answerable: bool


def make_answerable(rng: random.Random, qid: int) -> QAExample:
    obj = rng.choice(["shoe", "bag", "shirt", "watch", "phone"])  # e-commerce-ish objects
    color = rng.choice(["red", "blue", "green", "black", "white"])
    ctx = f"IMAGE: contains a {color} {obj}."
    q = f"What color is the {obj}?"
    return QAExample(qid=qid, question=q, context=ctx, answer=color, is_answerable=True)


def make_unanswerable(rng: random.Random, qid: int) -> QAExample:
    ex = make_answerable(rng, qid)
    mode = rng.choice(["missing_evidence", "wrong_object"])
    if mode == "missing_evidence":
        return QAExample(
            qid=qid,
            question=ex.question,
            context="IMAGE: evidence removed.",
            answer="<abstain>",
            is_answerable=False,
        )
    other = rng.choice(["shoe", "bag", "shirt", "watch", "phone"])
    while other in ex.question:
        other = rng.choice(["shoe", "bag", "shirt", "watch", "phone"])
    q = f"What color is the {other}?"
    return QAExample(qid=qid, question=q, context=ex.context, answer="<abstain>", is_answerable=False)


def build_dataset(n: int = 400, seed: int = 42) -> list[QAExample]:
    rng = random.Random(seed)
    out: list[QAExample] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(make_answerable(rng, i))
        else:
            out.append(make_unanswerable(rng, i))
    rng.shuffle(out)
    return out
