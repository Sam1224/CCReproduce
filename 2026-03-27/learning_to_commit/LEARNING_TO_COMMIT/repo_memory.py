from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RepoMemory:
    # A tiny stand-in for “online repository memory”.
    tests_required: bool
    docs_required: bool
    prefer_small_diff: bool

    def to_vec(self) -> List[float]:
        return [
            1.0 if self.tests_required else 0.0,
            1.0 if self.docs_required else 0.0,
            1.0 if self.prefer_small_diff else 0.0,
        ]


def memory_from_json(obj: Dict) -> RepoMemory:
    return RepoMemory(
        tests_required=bool(obj["tests_required"]),
        docs_required=bool(obj["docs_required"]),
        prefer_small_diff=bool(obj["prefer_small_diff"]),
    )
