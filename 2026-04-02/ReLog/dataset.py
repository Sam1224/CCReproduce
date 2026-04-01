from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LogGenExample:
    code: str
    runtime_trace: str
    target_level: int
    target_template: str


TEMPLATE_POOL = [
    "Entering {func}",
    "Exiting {func} with result={result}",
    "{func} args={args}",
    "{func} state: x={x} y={y}",
    "Error in {func}: {error}",
]


def load_toy_dataset() -> list[LogGenExample]:
    """A tiny dataset to demonstrate the ReLog pipeline.

    ReLog is an LLM-driven runtime-feedback framework; here we provide a minimal,
    fully runnable proxy dataset for training a small model that selects logging
    templates and log levels.
    """

    return [
        LogGenExample(
            code=(
                "def add(x, y):\n"
                "    return x + y\n"
            ),
            runtime_trace="call:add x=1 y=2 -> result=3",
            target_level=1,
            target_template="{func} state: x={x} y={y}",
        ),
        LogGenExample(
            code=(
                "def div(x, y):\n"
                "    return x / y\n"
            ),
            runtime_trace="call:div x=1 y=0 -> error=ZeroDivisionError",
            target_level=3,
            target_template="Error in {func}: {error}",
        ),
        LogGenExample(
            code=(
                "def checkout(cart_total, coupon):\n"
                "    if coupon: cart_total *= 0.9\n"
                "    return cart_total\n"
            ),
            runtime_trace="call:checkout cart_total=100 coupon=True -> result=90",
            target_level=1,
            target_template="{func} args={args}",
        ),
    ]


def iter_vocab(examples: Iterable[LogGenExample]) -> list[str]:
    vocab = {"<pad>", "<unk>"}
    for example in examples:
        vocab.update(tokenize(example.code))
        vocab.update(tokenize(example.runtime_trace))
    vocab.update(tokenize(" ".join(TEMPLATE_POOL)))
    return sorted(vocab)


def tokenize(text: str) -> list[str]:
    return [t for t in (text or "").replace("\n", " ").replace("/", " ").split(" ") if t]
