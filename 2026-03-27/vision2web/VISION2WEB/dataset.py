from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from spec import WebSpec, spec_from_json


@dataclass
class Task:
    tid: str
    spec: WebSpec
    label: int  # template id


def load_tasks(path: str) -> List[Task]:
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks.append(Task(tid=str(obj["id"]), spec=spec_from_json(obj["spec"]), label=int(obj["label"])))
    return tasks
