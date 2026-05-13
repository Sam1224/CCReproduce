from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NeedleSample:
    key: str
    value: str
    prompt: str


def build_needle_prompt(
    *,
    key: str,
    value: str,
    target_tokens: int,
    tokenizer,
    filler_sentence: str = "This is a filler sentence about shopping, products, and creators.",
) -> NeedleSample:
    """Build a long prompt with an inserted key/value pair and a question at the end.

    This is a toy dataset generator. The paper uses PG-19 and 5-digit values; we keep the same
    *interface* but generate synthetic filler so the script is runnable.
    """

    needle = f"\nMAGIC_KEY: {key}\nMAGIC_VALUE: {value}\n"
    question = (
        f"\nQuestion: Earlier in the document, what was the MAGIC_VALUE associated with {key}? "
        "Reply with only the value.\nAnswer:"
    )

    chunks = ["Document:\n"]
    chunks.append(filler_sentence + "\n")

    def current_len() -> int:
        return len(tokenizer("".join(chunks) + needle + question, return_tensors=None)["input_ids"])

    while current_len() < max(64, target_tokens - 64):
        chunks.append(filler_sentence + "\n")

    mid = len(chunks) // 2
    chunks.insert(mid, needle)

    prompt = "".join(chunks) + question
    return NeedleSample(key=key, value=value, prompt=prompt)
