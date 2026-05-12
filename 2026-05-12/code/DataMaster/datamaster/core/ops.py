from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .pool import DataPool
from .state import DataState, Record


@dataclass(frozen=True)
class AddSourceOp:
    source_name: str

    def name(self) -> str:
        return f"add_source:{self.source_name}"

    def apply(self, state: DataState, pool: DataPool) -> DataState:
        next_state = state.clone()
        next_state.records.extend(pool.get(self.source_name))
        next_state.sources.append(self.source_name)
        next_state.ops.append(self.name())
        return next_state


def op_dedup(records: list[Record]) -> list[Record]:
    seen: set[str] = set()
    out: list[Record] = []
    for r in records:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def op_filter_min_tokens(min_tokens: int) -> Callable[[list[Record]], list[Record]]:
    def _fn(records: list[Record]) -> list[Record]:
        out: list[Record] = []
        for r in records:
            if len(r["text"].split()) >= min_tokens:
                out.append(r)
        return out

    return _fn


_VIOLATION_PATTERNS = [
    r"free\s+money",
    r"100%\s+guarantee",
    r"counterfeit",
    r"scam",
    r"fake\s+brand",
    r"click\s+this\s+link",
]


def heuristic_label(text: str) -> int:
    t = text.lower()
    for p in _VIOLATION_PATTERNS:
        if re.search(p, t):
            return 1
    return 0


def op_drop_label_conflicts(records: list[Record]) -> list[Record]:
    out: list[Record] = []
    for r in records:
        h = heuristic_label(r["text"])
        if h != int(r["label"]):
            continue
        out.append(r)
    return out
