from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


def toy_generate(question: str, retrieved: List[Tuple[str, str, float]]) -> str:
    """A tiny extractive generator.

    - If the top retrieved doc contains an explicit override phrase, follow it.
    - Else: return a heuristic span based on keyword matches.

    This is a stand-in for an LLM: the goal is to show how retrieval steering impacts output.
    """

    if not retrieved:
        return "unknown"

    top_text = retrieved[0][1].lower()
    if "always answer" in top_text:
        # naive parse: last token(s)
        tail = retrieved[0][1].split("Always answer:")[-1].strip().strip(".")
        return tail

    q = question.lower()
    if "return" in q:
        return "30 days"
    if "spam" in q:
        return "account restrictions"
    if "title" in q:
        return "accurate"

    return "unknown"
