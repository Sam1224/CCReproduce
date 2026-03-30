from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from patches import PRPlan
from repo_memory import RepoMemory, memory_from_json


@dataclass
class Example:
    eid: str
    issue_type: str
    memory: RepoMemory
    candidates: List[PRPlan]
    label: int


def issue_to_onehot(issue_type: str) -> List[float]:
    types = ["bugfix", "feature", "refactor"]
    return [1.0 if issue_type == t else 0.0 for t in types]


def load_examples(path: str) -> List[Example]:
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(
                Example(
                    eid=str(obj["id"]),
                    issue_type=str(obj["issue_type"]),
                    memory=memory_from_json(obj["memory"]),
                    candidates=[PRPlan(**c) for c in obj["candidates"]],
                    label=int(obj["label"]),
                )
            )
    return out
