from __future__ import annotations

from dataclasses import dataclass
from typing import List

from repo_memory import RepoMemory


@dataclass
class PRPlan:
    # This is a proxy for a real PR diff.
    has_tests: bool
    updates_docs: bool
    diff_lines: int

    def to_vec(self) -> List[float]:
        return [
            1.0 if self.has_tests else 0.0,
            1.0 if self.updates_docs else 0.0,
            float(self.diff_lines) / 200.0,
        ]


def acceptance_oracle(memory: RepoMemory, pr: PRPlan) -> bool:
    if memory.tests_required and (not pr.has_tests):
        return False
    if memory.docs_required and (not pr.updates_docs):
        return False
    if memory.prefer_small_diff and pr.diff_lines > 120:
        return False
    return True


def score_oracle(memory: RepoMemory, pr: PRPlan) -> float:
    # A soft score: accepted PRs get higher score, with a mild preference for smaller diffs.
    ok = acceptance_oracle(memory, pr)
    base = 1.0 if ok else 0.0
    size_pen = min(1.0, pr.diff_lines / 200.0)
    return base - 0.2 * size_pen
